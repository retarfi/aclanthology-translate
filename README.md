# aclanthology-translate
Translate abstracts on ACL anthology into Japanese

## Usage
For example:
```sh
OPENAI_API_KEY=<your_openai_key> python main.py https://aclanthology.org/events/acl-2022/ \
--save_dir outputs/ \
--model gpt-3.5-turbo
```
You can choose volumes with indecies like `1,2`.  
Output file would be generated as outputs/2022acl-long_gpt-3.5-turbo.jsonl.  
Each jsonl file has following keys:

- title
- url
- abstract
- vol-title
- jtitle: Japanese translated title
- jabstract: Japanese translated abstract
