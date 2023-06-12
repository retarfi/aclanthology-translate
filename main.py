import argparse
import os
import json
import urllib.parse
from functools import partial
from typing import Dict, List, Optional, Set, Tuple

import openai
import requests
from bs4 import BeautifulSoup, Tag
from requests import Response
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm as std_tqdm

tqdm = partial(std_tqdm, dynamic_ncols=True)


def extract_volumes(soup: BeautifulSoup) -> List[Tuple[str, str]]:
    lst_vol: List[Tag] = (
        soup.find("div", class_="card-body")
        .find_all("ul", class_="list-pl-responsive")[-1]
        .find_all("li")
    )
    lst_tpl_vol: List[Tuple[str, str]] = list(
        map(lambda t: (t.find("a").text, t.find("a")["href"].lstrip("#")), lst_vol)
    )
    while True:
        print("Choose indices from following volumes (ex. 0,3,5):")
        for i, (title, _) in enumerate(lst_tpl_vol):
            print(f"{i}: {title}")
        s: str = input("Indices: ")
        lst_idx: List[str] = s.split(",")
        if not all(map(lambda x: x.isdecimal(), lst_idx)):
            print("Got invalid strings")
            continue
        set_input: Set[int] = set(map(int, lst_idx))
        set_avail: Set[int] = set(range(len(lst_tpl_vol)))
        if len(set_input - set_avail) > 0:
            set_invalid: Set[int] = set_input - set_avail
            print(f"Got invalid numbers: {set_invalid}")
            continue
        break
    lst_tpl_vol = [x for i, x in enumerate(lst_tpl_vol) if i in set_input]
    return lst_tpl_vol


def extract_paper_detail(url: str, vol_title: Optional[str] = None) -> Dict[str, str]:
    res: Response = requests.get(url)
    soup: BeautifulSoup = BeautifulSoup(res.text, "lxml")
    details: Optional[Tag] = soup.find("div", class_="row acl-paper-details")
    abstract: str
    if details is not None:
        abstract = details.find("span").text
    else:
        abstract = ""
    result: Dict[str, str] = {
        "title": soup.find("h2", id="title").text,
        "url": url,
        "abstract": abstract,
        "vol-title": vol_title,
    }
    return result


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def translate_to_japanese(text: str, model: str = "gpt-3.5-turbo") -> str:
    fmt_prompt: str = (
        "Following sentences are the abstract of the paper about natural language "
        "processing. Translate them into Japanese:\n\n{}"
    )
    completion = completion_with_backoff(
        model=model,
        messages=[{"role": "user", "content": fmt_prompt.format(text)}],
        temperature=0,
    )
    return completion.choices[0].message["content"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Translate abstracts on ACL anthology. "
            "Enviromental variable OPENAI_API_KEY must be set."
        )
    )
    parser.add_argument(
        "url",
        help=(
            "URL of event page on ACL anthology. It must be like "
            "https://aclanthology.org/events/acl-2022/"
        ),
    )
    parser.add_argument("--save-dir", help="Save directory", default="outputs/")
    parser.add_argument(
        "--model",
        help="OpenAI model",
        default="gpt-3.5-turbo",
        choices=["gpt-3.5-turbo", "gpt-4"],
    )

    args = parser.parse_args()
    base_url: str = "https://aclanthology.org"

    api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    assert api_key is not None, "Specify environment variable OPENAI_API_KEY"
    openai.api_key = api_key
    # confirm api key is valid
    model: str = args.model
    assert model in list(
        map(lambda x: x["id"], openai.Model.list()["data"])
    ), f"{model} is not available for OpenAI model"

    os.makedirs(args.save_dir, exist_ok=True)

    res: Response = requests.get(args.url)
    soup: BeautifulSoup = BeautifulSoup(res.text, "lxml")
    lst_tpl_vol: List[Tuple[str, str]] = extract_volumes(soup)
    for vol_title, vol_id in tqdm(lst_tpl_vol):
        lst_data: List[Dict[str, str]]
        path_jsonl = os.path.join(args.save_dir, f"{vol_id}_{model}.jsonl")
        if os.path.isfile(path_jsonl):
            with open(path_jsonl, "r") as f:
                lst_data = [json.loads(l) for l in f.readlines()]
        else:
            lst_data = []
        set_finished_url: Set[str] = set([x["url"] for x in lst_data])
        lst_papers: List[Tag] = soup.find("div", id=vol_id).find_all("p")
        for tag in tqdm(lst_papers, leave=False):
            directory: str = tag.find("strong").find("a")["href"]
            paper_url: str = urllib.parse.urljoin(base_url, directory)
            if paper_url in set_finished_url:
                continue
            paper_data: Dict[str, str] = extract_paper_detail(
                paper_url, vol_title=vol_title
            )
            paper_data["jtitle"] = translate_to_japanese(paper_data["title"], model)
            if paper_data["abstract"] == "":
                paper_data["jabstract"] = ""
            else:
                paper_data["jabstract"] = translate_to_japanese(
                    paper_data["abstract"], model
                )
            lst_data.append(paper_data)
            with open(path_jsonl, "a") as f:
                # f.writelines([json.dumps(l, ensure_ascii=False) for l in lst_data])
                f.write(json.dumps(paper_data, ensure_ascii=False) + "\n")
