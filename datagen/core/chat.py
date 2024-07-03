from pathlib import Path
from typing import Optional
import traceback

import base64
import cv2
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


from .config import DatagenConfig
from .img_utils import cv2_resize
from .types import LLMInput


# You work with videos of people doing squats.
# You are given a question about the video and set of subsequent frames and need to answer the question.
# Answer to the question, as given by an experienced fitness coach.
# Don't be shy to point out the flaws.
# Critique or feedback on exercise performance.
# Do not make comments about anything besides exercise performance.
# Be concise but don't leave out details.

system_prompt_default = """You're a data annotator."""

def ask(llm_input: LLMInput, config: DatagenConfig):
    """Generic LLM query, can have prompt, system prompt, and images

    Args:
        config (DatagenConfig): _description_
        imgs (Optional[np.array], optional): _description_. Defaults to None.
        human_prompt (Optional[str | List[str]], optional): _description_. Defaults to None.
        schema (Optional[BaseModel], optional): _description_. Defaults to None.
        system_prompt (Optional[str], optional): _description_. Defaults to system_prompt_default.
        ntiles (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    messages = []
    if llm_input.system_prompt:
        messages.append(SystemMessage(llm_input.system_prompt))
    if llm_input.human_prompt:
        if type(llm_input.human_prompt) is str:
            messages.append(HumanMessage(llm_input.human_prompt))
        else:
            for p in llm_input.human_prompt:
                messages.append(HumanMessage(p))
    if llm_input._imgs:
        imgs = [cv2_resize(f[:,:,::-1], 512*llm_input.ntiles, 512*llm_input.ntiles) for f in llm_input._imgs]
        imgs_b64 = [base64.b64encode(cv2.imencode('.jpg', f)[1]).decode('utf-8') for f in imgs]
        messages.append(HumanMessage([{
            'type': 'image_url',
            'image_url': {
                'url': f"data:image/jpeg;base64,{img_b64}",
            },
        } for img_b64 in imgs_b64]))
    prompt = ChatPromptTemplate.from_messages(messages)
    chain = prompt | (config.model.with_structured_output(schema=llm_input.output_schema) if llm_input.output_schema else config.model)
    try:
        out = chain.invoke({})
        return out if llm_input.output_schema else out.content
    except Exception as e:
        print(e)
        # print(traceback.format_exc())
        return None