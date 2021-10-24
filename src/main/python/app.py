import logging
import os
from typing import List

import torch
from fastapi import FastAPI
from inference_management import *
from pydantic import BaseModel, root_validator

logging.getLogger().setLevel("DEBUG")

logging.debug(
    "{} Inference Server - Visible CUDA Capable GPUs.".format(torch.cuda.device_count())
)


comprehension_engine = ZeroComprehensionEngine()


class Item(BaseModel):
    input_string: str
    label_options: List[str]

    @root_validator(pre=True)
    def check_input_string(cls, values):
        assert (
            "input_string" in values
        ), "input_string should be included as this is a text inference engine."
        return values

    @root_validator(pre=True)
    def check_label_options(cls, values):
        assert (
            "label_options" in values
        ), "label_options should be included as this is a text inference engine and it needs options to choose from."
        return values


app = FastAPI()


@app.get("/healthcheck")
async def read_main():
    return 200


@app.post("/predict")
async def create_item(item: Item):

    logging.debug(item.input_string)

    logging.debug(item.label_options)

    return comprehension_engine.free_comprehend(item.input_string, item.label_options)
