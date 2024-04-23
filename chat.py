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
from llama.generation import sample_top_p


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


@torch.inference_mode()
def generate_print(
    generator,
    prompt_tokens: List[int],
    max_gen_len: int,
    temperature: float = 0.6,
    top_p: float = 0.9,
) -> List[str]:
    """
    Args:
        prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
        max_gen_len (int): Maximum length of the generated text sequence.
        temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
        top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
    """
    params = generator.model.params
    prompt_len = len(prompt_tokens)
    total_len = min(params.max_seq_len, max_gen_len + prompt_len)

    pad_id = generator.tokenizer.pad_id
    tokens = torch.full((1, total_len,), pad_id, dtype=torch.long, device="cuda")
    tokens[:, : len(prompt_tokens)] = torch.tensor(prompt_tokens, dtype=torch.long, device="cuda")

    prev_pos = 0
    eos_reached = torch.tensor([False], device="cuda")
    input_text_mask = tokens != pad_id
    if prompt_len == total_len:
        logits = generator.model.forward(tokens, prev_pos)

    stop_tokens = torch.tensor(list(generator.tokenizer.stop_tokens))
    reply = []
    for cur_pos in range(prompt_len, total_len):
        logits = generator.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)

        next_token = next_token.reshape(-1)
        # only replace token if prompt has already been generated
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        tokens[:, cur_pos] = next_token
        eos_reached |= (~input_text_mask[:, cur_pos]) & (
            torch.isin(next_token, stop_tokens)
        )
        prev_pos = cur_pos
        if all(eos_reached):
            break
        word = generator.tokenizer.decode(next_token.tolist())
        print(word, end="", flush=True)
        reply.append(word)
    return reply


def main(
    ckpt_dir: str = "Meta-Llama-3-8B-Instruct",
    tokenizer_path: str = "Meta-Llama-3-8B-Instruct/tokenizer.model",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 6000,
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
        max_batch_size=1,
    )
    print("context length:", max_seq_len)
    print(generator.model)

    phrase_quit, phrase_new, phrase_back, phrase_text, phrase_pdf, phrase_url = "#QUIT", "#NEW", "#BACK", "#TEXT", "#PDF", "#URL"
    preset = [
        {"role": "system", "content": "Think in steps before answering. Give straightforward and concise answers. If you do not know the answer, reply with I do not know."}
    ]
    print("stop inference server:  %s" % phrase_quit)
    print("start new conversation: %s" % phrase_new)
    print("roll back one query:    %s" % phrase_back)
    print("read from text file:    %s <text-file-path>" % phrase_text)
    print("read from PDF file:     %s <PDF-file-path>" % phrase_pdf)
    print("read from webpage:      %s <URL>" % phrase_url)

    while True:
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
                elif query.lower().startswith(phrase_back.lower()):
                    if len(context) > 1:
                        context.pop(-1)
                        context.pop(-1)
                    print("🛑 > rolling back")
                    continue

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

            context.append({"role": "user", "content": query})
            torch.cuda.empty_cache()

            prompt_tokens = generator.formatter.encode_dialog_prompt(context)
            if len(prompt_tokens) >= generator.model.params.max_seq_len:
                print("🛑 > maximum context length reached, rolling back")
                context.pop(-1)
                continue

            print("🦙 > ", end="", flush=True)
            time0 = time.perf_counter()
            reply = generate_print(
                generator,
                prompt_tokens=prompt_tokens,
                max_gen_len=generator.model.params.max_seq_len - 1,
                temperature=temperature,
                top_p=top_p,
            )
            time_gen = time.perf_counter() - time0
            print(" [%d/%.2f = %.2f tokens/s]" % (len(reply), time_gen, len(reply) / time_gen))
            context.append({"role": "assistant", "content": "".join(reply)})


if __name__ == "__main__":
    fire.Fire(main)
