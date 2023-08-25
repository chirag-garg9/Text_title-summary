import os
import urllib.request as request
import zipfile
from pathlib import Path
from src.TextsummerizeProject.logging import logger
from src.TextsummerizeProject.utils.common import get_size
from src.TextsummerizeProject.entities import DataIngestionconfig
from src.TextsummerizeProject.entities import Datavalidationconfig
from src.TextsummerizeProject.entities import Datatransformationconfig
from src.TextsummerizeProject.entities import ModelTrainerConfig
from src.TextsummerizeProject.entities import ModelEvaluationConfig
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import datasets
from datasets import load_metric
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
import pandas as pd
import re
import pickle

class DataIngestion:
    def __init__(self, config: DataIngestionconfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, header = request.urlretrieve(
                url = self.config.source_url,
                filename= self.config.local_data_file
            )
            logger.info(f"{filename} download with following info: \n{header}")
        else:
            logger.info(f"{self.config.local_data_file} already exists of size {get_size(Path(self.config.local_data_file))}")
    
    def extract_file(self):
        '''Extracts the ZIP file'''
        unzip_path = Path(self.config.unzip_directory)
        with zipfile.ZipFile(self.config.local_data_file,'r') as unzip_file:
            unzip_file.extractall(unzip_path)


class Datavalidation:
    def __init__(self,config:Datavalidationconfig):
        self.config = config

    def validate(self)->bool:
        try:
            validation_status = None    
            all_files = os.listdir(os.path.join('artifacts','data_ingestion'))

            for file in all_files:
                if file not in self.config.all_required_files:
                    validation_status = False
                    with open(self.config.status_file,'w') as f:
                        f.write(f'validation_status: {validation_status}')
                else:
                    validation_status = True
                    with open(self.config.status_file,'w') as f:
                        f.write(f'validation_status: {validation_status}')

        except Exception as e:
            raise e
    
class Datatransformation:
    def __init__(self, config:Datatransformationconfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)


    def covert_to_features(self,source,target):
        input_encoding = self.tokenizer(source,max_length=512,truncation=True,padding=True,is_split_into_words=True)

        with self.tokenizer.as_target_tokenizer():
            target_encoding = self.tokenizer(target,max_length=512,truncation=True,padding=True,is_split_into_words=True)

        return{
            'input_ids': input_encoding['input_ids'],
            'attention_mask': input_encoding['attention_mask'],
            'labels': target_encoding['input_ids']
            }

    def convert(self):
        sci_dataset = pd.read_json(self.config.data_path,lines=True)
        sci_dataset.dropna()
        # sci_dataset['source']=sci_dataset['source'].apply(lambda x: re.sub('<>:/\\|\?\*','',x))
        # sci_dataset['test']=sci_dataset['test'].apply(lambda x: re.sub('<>:/\\|\?\*','',x))
        x_train = list(sci_dataset['source'])
        y_train = list(sci_dataset['target'])
        sci_dataset_test = pd.read_json(self.config.data_path_test,lines=True)
        sci_dataset_test.dropna()
        # sci_dataset_test['source']=sci_dataset_test['source'].apply(lambda x: re.sub('<>:/\\|?*','',x))
        # sci_dataset_test['test']=sci_dataset_test['test'].apply(lambda x: re.sub('<>:/\\|?*','',x))
        x_test = list(sci_dataset['source'])
        y_test = list(sci_dataset['target'])
        dataset_pt = self.covert_to_features(x_train,y_train)
        dataset_pt_test = self.covert_to_features(x_train,y_train) 
        path,_ = os.path.split(self.config.data_path)
        path_test,_ = os.path.split(self.config.data_path_test)
        path = os.path.join(self.config.root_dir,'train.pkl')
        path_test = os.path.join(self.config.root_dir,'test.pkl')
        with open(path, 'wb') as f:
            pickle.dump(dataset_pt, f)
        with open(path_test, 'wb') as f:
            pickle.dump(dataset_pt_test, f)

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    
    def train(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model_scibert = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt)
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_scibert)
        
        #loading data 
        path_train = os.path.join(self.config.data_path,'train.pkl')
        path_test = os.path.join(self.config.data_path,'test.pkl')
        trainfile = pickle.load(open(path_train,'rb'))
        testfile = pickle.load(open(path_test,'rb'))
        # x_train ={'input_ids':trainfile['input_ids'], 'attention_mask':trainfile['attention_mask']}
        # y_train ={'labels':trainfile['labels']}
        # x_test ={'input_ids':testfile['input_ids'], 'attention_mask':testfile['attention_mask']}
        # y_test ={'labels':testfile['labels']}
        # trainset = tf.data.Dataset.from_tensor_slices((dict(x_train),y_train))
        # testset = tf.data.Dataset.from_tensor_slices((dict(x_test),y_test))

        trainset = datasets.Dataset.from_pandas(pd.DataFrame(data=trainfile)[:1200])
        testset = datasets.Dataset.from_pandas(pd.DataFrame(data=testfile)[:300])

        trainer_args = TrainingArguments(
            output_dir=self.config.root_dir, 
            learning_rate=0.004,
            num_train_epochs=1, warmup_steps=500,
            per_device_train_batch_size=1,
            weight_decay=0.01, logging_steps=10,
            evaluation_strategy='steps', save_steps=1e6,
            gradient_accumulation_steps=16
        ) 

        trainer = Trainer(model=model_scibert, args=trainer_args,
                  tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                  train_dataset=trainset,eval_dataset=testset)
        
        trainer.train()

        ## Save model
        model_scibert.save_pretrained(os.path.join(self.config.root_dir,"scibert-ML/DL-model"))
        ## Save tokenizer
        tokenizer.save_pretrained(os.path.join(self.config.root_dir,"tokenizer"))


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config


    
    def generate_batch_sized_chunks(self,list_of_elements, batch_size):
        """split the dataset into smaller batches that we can process simultaneously
        Yield successive batch-sized chunks from list_of_elements."""
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    
    def calculate_metric_on_test_ds(self,dataset, metric, model, tokenizer, 
                               batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu", 
                               column_text="article", 
                               column_summary="highlights"):
        dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=dataset))
        article_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
        target_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))
        
        for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches), total=len(article_batches)):
            
            inputs = tokenizer(article_batch, max_length=4096,  truncation=True, 
                            padding="max_length", return_tensors="pt",is_split_into_words=True)
            
            summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                            attention_mask=inputs["attention_mask"].to(device), 
                            length_penalty=0.8, num_beams=8, max_length=250)
            ''' parameter for length penalty ensures that the model does not generate sequences that are too long. '''
            
            # Finally, we decode the generated texts, 
            # replace the  token, and add the decoded texts with the references to the metric.
            decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, 
                                    clean_up_tokenization_spaces=True) 
                                for s in summaries]      
            
            decoded_summaries = [d.replace("", " ") for d in decoded_summaries]
            
            
            metric.add_batch(predictions=decoded_summaries, references=target_batch)
            
        #  Finally compute and return the ROUGE scores.
        score = metric.compute()
        return score


    def evaluate(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        scibert = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)
       
        #loading data 
        path_test = os.path.join(self.config.data_path)
        testfile = pd.read_json(path_test,lines=True)


        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
  
        rouge_metric = load_metric('rouge')

        score = self.calculate_metric_on_test_ds(
        testfile.iloc[:10], rouge_metric, scibert, tokenizer, batch_size = 2, column_text = 'source', column_summary= 'target'
        )

        rouge_dict = dict((rn, score[rn].mid.fmeasure ) for rn in rouge_names )

        df = pd.DataFrame(rouge_dict, index = ['scibert'] )
        df.to_csv(self.config.metric_file_name, index=False)