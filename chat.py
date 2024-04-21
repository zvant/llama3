# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import os
import copy
import time
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
import pdfplumber
import fire

import torch
from llama import Dialog, Llama


def read_text_file(filename):
    if not os.access(filename, os.R_OK):
        print("📂 > cannot read %s" % filename)
        return None
    try:
        with open(filename, "r") as fp:
            text = str(fp.read())
    except:
        print("📂 > cannot read %s" % filename)
        return None
    print("📂 > read text from %s" % filename)
    return text


def parse_pdf_file(filename):
    if not os.access(filename, os.R_OK):
        print("📂 > cannot read %s" % filename)
        return None
    try:
        text = []
        with pdfplumber.open(filename) as fp:
            for page in fp.pages:
                text.append(page.extract_text())
        text = "\n\n".join(text)
    except:
        print("📂 > cannot read %s" % filename)
        return None
    print("📂 > parsed content from %s" % filename)
    return text


def fetch_url(url):
    try:
        res = requests.get(url, timeout=3)
    except:
        print("🌐 > cannot fetch %s" % url)
        return None
    html_content = BeautifulSoup(res.content, "html.parser")
    text = html_content.find_all(string=True)
    text = "\n\n".join(text)
    print("🌐 > fetched and parsed content from %s" % url)
    return text


def main(
    ckpt_dir: str = "Meta-Llama-3-8B-Instruct",
    tokenizer_path: str = "Meta-Llama-3-8B-Instruct/tokenizer.model",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 8000,
    max_batch_size: int = 1,
    max_gen_len: Optional[int] = None,
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    print("context length:", max_seq_len)
    print(generator.model)

    phrase_quit, phrase_new, phrase_text, phrase_pdf, phrase_url = "#QUIT", "#NEW", "#TEXT", "#PDF", "#URL"
    preset = [
        {"role": "system", "content": "Think in steps before answering. Give straightforward and concise answers. If you do not know the answer, reply with I do not know."}
    ]
    print("stop inference server:  %s" % phrase_quit)
    print("start new conversation: %s" % phrase_new)
    print("read from text file:    %s <text-file-path>" % phrase_text)
    print("read from PDF file:     %s <PDF-file-path>" % phrase_pdf)
    print("read from webpage:      %s <URL>" % phrase_url)

    while True:
        torch.cuda.empty_cache()
        print("============== start new 🦙 conversation ==============")
        context = copy.deepcopy(preset)
        context[0]["content"] += ' Now is %s.' % time.strftime('%H:%M %Z, %A, %Y %b %d')
        while True:
            query = str(input("🗣️ > ")).strip()
            if len(query) < 1:
                continue
            if query[0] == "#":
                if query.lower().startswith(phrase_quit.lower()):
                    return
                elif query.lower().startswith(phrase_new.lower()):
                    break

                elif query.lower().startswith(phrase_text.lower()):
                    filename = query[len(phrase_text):].strip()
                    text = read_text_file(filename)
                    if text is None:
                        continue
                    query = "Read the following content loaded from the text file %s, and answer my questions about it. Give reference in the original content if possible. Start of the content:\n%s" % (filename, text)

                elif query.lower().startswith(phrase_pdf.lower()):
                    filename = query[len(phrase_pdf):].strip()
                    text = parse_pdf_file(filename)
                    if text is None:
                        continue
                    query = "Read the following content parsed from the PDF file %s, and answer my questions about it. Give reference in the original content if possible. Start of the content:\n%s" % (filename, text)

                elif query.lower().startswith(phrase_url.lower()):
                    url = query[len(phrase_url):].strip()
                    text = fetch_url(url)
                    if text is None:
                        continue
                    query = "Read the following content fetched from the URL %s, and answer my questions about it. Give reference in the original content if possible. Start of the content:\n%s" % (url, text)

            # print(query)
            context.append({"role": "user", "content": query})

            try:
                reply = generator.chat_completion(
                    [context],
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                )[0]
            except Exception as e:
                print("🛑 > exception: %s, rolling back" % str(e))
                context.pop(-1)
                continue

            tokens_count, time_gen = reply["tokens_count"], reply["time_gen"]
            reply = reply["generation"]
            assert reply["role"] == "assistant"
            print("🦙 > %s [%d/%.2f = %.2f tokens/s]" % (reply["content"], tokens_count, time_gen, tokens_count / time_gen))
            context.append(reply)


if __name__ == "__main__":
    fire.Fire(main)
