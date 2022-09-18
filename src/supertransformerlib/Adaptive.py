"""

A collection of functions designed to
aid in the design of more intelligent
algorithms.

"""
import dataclasses
from typing import Optional, Union, List

import torch
from torch import nn

from . import Core
from . import Attention

@dataclasses.dataclass
class State:

    @property
    def exhaustion_mask(self):
        return self.stamina == 0

    tensor: torch.Tensor
    stamina: torch.Tensor


class Investigation(Core.KernelSpace, Core.Utility):
    """
    The investigation layer. Performs queries
    of background and preprocessed resources.

    Investigating Proceeds as:

    * State_History -> get -> Current State
    * Current State-> Question
    * Question -> QA -> Answer
    * Last State, Question, Answer -> Concat -> Investigative_Summary
    """
    def __init__(self,
                 d_model: int,
                 d_engine: int,
                 heads : int,
                 question_engine: nn.Module,
                 parallelization: Optional[Union[torch.Tensor, List[int], int]] = None,
                 dynamics: Optional[int] = None):

        super().__init__()
        self.Question_Generator = Core.Linear(d_model, [heads, d_engine], parallelization, dynamics)
        self.Engine = question_engine
        self.Answer_Converter = Core.Linear([heads, d_engine], d_model, parallelization, dynamics)

    def forward(self, status: State):

        status_tensor = status.tensor
        current_state = status_tensor[..., -1, :]


        question = self.Question_Generator(status_tensor)
        question = question.masked_fill(status.exhaustion_mask, 0)


        answer = self.Engine(question)
        answer = self.Answer_Converter(answer)
        answer = answer.masked_fill(status.exhaustion_mask)

        output = torch.concat([current_state, question, answer], dim=-1)
        return output

class Revision:
    """
    The revision layer. Takes in an Investigative_Summary, State, and Stamina tensor.
    Returns State and Stamina Tensor

    Proceeds as:

    Investigative_Summary -> Linear -> Output_Query, Cleanup_Query,  State_Update,
    Output_Query, Cleanup_Query, State_History, State_Update -> Concat -> Composite State

    make stamina

    Composite State, Exhaustion -> Exhaustion Transformer -> Composite State, Exhaustion, Residuals
    Composite State -> Feedforward -> Composite State
    Composite State -> Split -> Output, Trash, State_History

    """

class APG(nn.Module):
    """
    Adaptive Planning Generator.
    Status Update:

    Investigating:

    * State_History -> Current State
    * Current state -> Question
    * Question -> QA -> Answer
    * Current_state, Question, Answer -> Concat -> Investigative_Summary

    Revision:

    Investigative_Summary -> Linear -> Output_Query, Cleanup_Query,  State_Update,
    Output_Query, Cleanup_Query, State_History, Answers -> Concat -> Composite State
    Composite State, Exhaustion -> Exhaustion Transformer -> Composite State, Exhaustion, Residuals
    Composite State -> Feedforward -> Composite State
    Composite State -> Split -> Output, Trash, State_History
    State_History -> Get -> State Update

    Reflecting

    State_Update -> Linear ->  Halting Prob
    Output_buffer.append(Output)
    if cumulative_halting_prob >= 1:
        halt
        return output, state, residuals.

    Maintence:

    if stamina == 0:
        prune







    * Update branch probabilities
    * Execute branches
    *   If halt:
    *       return state, exhaustion, residuals.
    *   elif discard:
            pass
    *   else:
    *       status update
    * Status_Update -> branch probs
    *   If branch: Execute. IF halt, escape
    *   If halt: Return
    * Status_Update, State_History -> Concat -> State_History
    * New_Exhaustion -> Exhaustion


    """
