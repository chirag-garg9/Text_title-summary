from src.TextsummerizeProject.pipeline.Data_transformation import datatransformationpipeline
from src.TextsummerizeProject.pipeline.Data_Validation import datavalidationpipeline
from src.TextsummerizeProject.pipeline.Data_injestion import dataingestionpipeline
from src.TextsummerizeProject.pipeline.Model_Evaluation import ModelEvaluationPipeline
from src.TextsummerizeProject.pipeline.Model_Trainer import ModelTrainerPipeline
from src.TextsummerizeProject.logging import logger


stage = 'Data_injestion'

try:
    logger.info('>>>>>>> stage {stage1} initiated\n'.format(stage1=stage))
    datainjestion = dataingestionpipeline()
    datainjestion.main()
    logger.info('>>>>>>> stage {stage1} completed\n'.format(stage1=stage))
except Exception as e:
    logger.exception(e)
    raise e


stage = 'Data_Validation'

try:
    logger.info('>>>>>>> stage {stage2} initiated\n'.format(stage2=stage))
    datainjestion = datavalidationpipeline()
    datainjestion.main()
    logger.info('>>>>>>> stage {stage2} completed\n'.format(stage2=stage))
except Exception as e:
    logger.exception(e)
    raise e

stage = 'Data_transformation'

try:
    logger.info('>>>>>>> stage {stage2} initiated\n'.format(stage2=stage))
    datainjestion = datatransformationpipeline()
    datainjestion.main()
    logger.info('>>>>>>> stage {stage2} completed\n'.format(stage2=stage))
except Exception as e:
    logger.exception(e)
    raise e

stage = 'ModelTrainingStage'

try:
    logger.info('>>>>>>> stage {stage2} initiated\n'.format(stage2=stage))
    datainjestion = ModelTrainerPipeline()
    datainjestion.main()
    logger.info('>>>>>>> stage {stage2} completed\n'.format(stage2=stage))
except Exception as e:
    logger.exception(e)
    raise e

stage = 'ModelEvaluationStage'

try:
    logger.info('>>>>>>> stage {stage2} initiated\n'.format(stage2=stage))
    datainjestion = ModelEvaluationPipeline()
    datainjestion.main()
    logger.info('>>>>>>> stage {stage2} completed\n'.format(stage2=stage))
except Exception as e:
    logger.exception(e)
    raise e