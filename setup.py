import setuptools

with open('Readme.md','w') as f:
    description = f.read()

version = '0.0.0'

Repo_name = 'Text-summerizer'
Author_name = 'chirag-garg9'
Author_git_url = 'https://github.com/chirag-garg9/Text-summerizer.git'
repo_src = 'TextsummerizeProject'
Author_email = 'chirag.garg.5293@gmail.com'

setuptools.setup(
    name=Repo_name,
    version=version,
    author=Author_name,
    author_email=Author_email,
    description='Text summerization site',
    long_description=description,
    long_description_content_type="text/markdown",
    url=Author_git_url,
    project_urls = { 'Bug tracker' : 'https://github.com/chirag-garg9/Text-summerizer.git/issues' },
    package_dir={"":'src'},
    packages=setuptools.find_packages(where='src'),)
    
