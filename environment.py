from enum import Enum
from typing import List

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class CardColor(str, Enum):
    RED = "Red"
    BLUE = "Blue"
    GREEN = "Green"
    YELLOW = "Yellow"
    WILD = "Wild"


class CardValue(str, Enum):
    ZERO = "0"
    ONE = "1"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    SKIP = "Skip"
    REVERSE = "Reverse"
    DRAW_TWO = "+2"
    STANDARD = "Standard"
    WILD_FOUR = "+4"


class Card(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    color: CardColor
    value: CardValue

    @model_validator(mode="after")
    def validate_logic(self) -> "Card":
        if self.color == CardColor.WILD and self.value not in [
            CardValue.STANDARD,
            CardValue.WILD_FOUR,
        ]:
            raise ValueError(
                f"Wild cards cannot have a standard value like '{self.value.value}'"
            )
        if self.color != CardColor.WILD and self.value in [
            CardValue.STANDARD,
            CardValue.WILD_FOUR,
        ]:
            raise ValueError(
                f"Colored cards cannot have a wild value like '{self.value.value}'"
            )
        return self

    def to_index(self) -> int:
        if self.color == CardColor.WILD:
            return 52 if self.value == CardValue.STANDARD else 53

        color_offsets = {
            CardColor.RED: 0,
            CardColor.BLUE: 1,
            CardColor.GREEN: 2,
            CardColor.YELLOW: 3,
        }
        value_offsets = {
            CardValue.ZERO: 0,
            CardValue.ONE: 1,
            CardValue.TWO: 2,
            CardValue.THREE: 3,
            CardValue.FOUR: 4,
            CardValue.FIVE: 5,
            CardValue.SIX: 6,
            CardValue.SEVEN: 7,
            CardValue.EIGHT: 8,
            CardValue.NINE: 9,
            CardValue.SKIP: 10,
            CardValue.REVERSE: 11,
            CardValue.DRAW_TWO: 12,
        }
        return (color_offsets[self.color] * 13) + value_offsets[self.value]

    @staticmethod
    def from_index(idx: int) -> "Card":
        if idx == 52:
            return Card(color=CardColor.WILD, value=CardValue.STANDARD)
        if idx == 53:
            return Card(color=CardColor.WILD, value=CardValue.WILD_FOUR)

        colors = [CardColor.RED, CardColor.BLUE, CardColor.GREEN, CardColor.YELLOW]
        values = [
            CardValue.ZERO,
            CardValue.ONE,
            CardValue.TWO,
            CardValue.THREE,
            CardValue.FOUR,
            CardValue.FIVE,
            CardValue.SIX,
            CardValue.SEVEN,
            CardValue.EIGHT,
            CardValue.NINE,
            CardValue.SKIP,
            CardValue.REVERSE,
            CardValue.DRAW_TWO,
        ]

        color_idx = idx // 13
        value_idx = idx % 13

        return Card(color=colors[color_idx], value=values[value_idx])

    def __rich__(self):

        color_tags = {
            CardColor.RED: "red",
            CardColor.BLUE: "blue",
            CardColor.GREEN: "green",
            CardColor.YELLOW: "bright_yellow",
            CardColor.WILD: "magenta",
        }

        tag = color_tags[self.color]

        return f"[{tag}][bold]{self.color} {self.value}[/bold][/{tag}]"

    def __str__(self) -> str:
        return f"{self.color} {self.value}"

    def __repr__(self):
        return f"{self.color} {self.value}"


class Deck(BaseModel):
    cards: List[Card] = Field(default_factory=list)

    def model_post_init(self, __context):
        standard_colors = [
            CardColor.RED,
            CardColor.BLUE,
            CardColor.GREEN,
            CardColor.YELLOW,
        ]
        standard_values = [
            CardValue.ZERO,
            CardValue.ONE,
            CardValue.TWO,
            CardValue.THREE,
            CardValue.FOUR,
            CardValue.FIVE,
            CardValue.SIX,
            CardValue.SEVEN,
            CardValue.EIGHT,
            CardValue.NINE,
            CardValue.SKIP,
            CardValue.REVERSE,
            CardValue.DRAW_TWO,
        ]

        for color in standard_colors:
            for value in standard_values:
                self.cards.append(Card(color=color, value=value))
                if value != CardValue.ZERO:
                    self.cards.append(Card(color=color, value=value))

        for _ in range(4):
            self.cards.append(Card(color=CardColor.WILD, value=CardValue.STANDARD))
            self.cards.append(Card(color=CardColor.WILD, value=CardValue.WILD_FOUR))

        self.shuffle()

    def shuffle(self) -> None:
        if not self.cards:
            return

        indices = np.random.permutation(len(self.cards))
        self.cards = [self.cards[i] for i in indices]

    def draw(self) -> Card:
        if not self.cards:
            self.model_post_init(None)
        return self.cards.pop()


class UnoEnv:
    def __init__(self) -> None:
        self.num_actions = 55
        self.base_colors = [
            CardColor.RED,
            CardColor.BLUE,
            CardColor.GREEN,
            CardColor.YELLOW,
        ]
        self.reset()

    def reset(self) -> np.ndarray:
        self.discard_pile_counts = np.zeros(54, dtype=np.float32)
        self.deck = Deck()

        self.agent_hand: List[Card] = []
        self.opponent_hand_size = 7

        for _ in range(7):
            self.agent_hand.append(self.deck.draw())
        for _ in range(7):
            self.deck.draw()

        self.top_card = self.deck.draw()
        self.discard_pile_counts[self.top_card.to_index()] += 1

        if self.top_card.color == CardColor.WILD:
            self.current_color = self.base_colors[np.random.randint(0, 4)]
        else:
            self.current_color = self.top_card.color

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        hand_vector = np.zeros(54, dtype=np.float32)
        for card in self.agent_hand:
            hand_vector[card.to_index()] += 1

        color_offsets = {
            CardColor.RED: 0,
            CardColor.BLUE: 1,
            CardColor.GREEN: 2,
            CardColor.YELLOW: 3,
        }
        color_one_hot = np.zeros(4, dtype=np.float32)

        if self.current_color in color_offsets:
            color_one_hot[color_offsets[self.current_color]] = 1.0

        return np.concatenate([
            hand_vector,
            self.discard_pile_counts,
            color_one_hot,
            np.array([self.opponent_hand_size], dtype=np.float32),
        ])

    def get_legal_mask(self) -> np.ndarray:
        mask = np.zeros(61, dtype=np.float32)
        has_legal_move = False

        for card in self.agent_hand:
            idx = card.to_index()
            if card.color == CardColor.WILD:
                if card.value == CardValue.STANDARD:
                    mask[52:56] = 1.0  # Actions 52, 53, 54, 55
                elif card.value == CardValue.WILD_FOUR:
                    mask[56:60] = 1.0  # Actions 56, 57, 58, 59
                has_legal_move = True
            elif card.color == self.current_color:
                mask[idx] = 1.0
                has_legal_move = True
            elif (
                self.top_card.color != CardColor.WILD
                and card.value == self.top_card.value
            ):
                mask[idx] = 1.0
                has_legal_move = True

        if not has_legal_move:
            mask[60] = 1.0  # Action 60 is now Draw

        return mask

    def _reshuffle_discard(self):
        # 1. Temporarily remove the top card from the counts
        self.discard_pile_counts[self.top_card.to_index()] -= 1

        # 2. Rebuild the physical deck based on the discard counts
        new_cards = []
        for idx, count in enumerate(self.discard_pile_counts):
            for _ in range(int(count)):
                new_cards.append(Card.from_index(idx))

        self.deck.cards = new_cards
        self.deck.shuffle()

        # 3. Reset the discard pile vector to contain ONLY the top card
        self.discard_pile_counts = np.zeros(54, dtype=np.float32)
        self.discard_pile_counts[self.top_card.to_index()] = 1.0

    def step(self, action_idx: int) -> tuple[np.ndarray, float, bool, np.ndarray]:
        if action_idx == 60:  # Draw Action
            if len(self.deck.cards) == 0:
                self._reshuffle_discard()
            if len(self.deck.cards) > 0:
                self.agent_hand.append(self.deck.draw())
        else:
            chosen_card = None
            chosen_color = None

            # Map action back to physical card index
            if 52 <= action_idx <= 55:
                target_idx = 52  # Standard Wild
                chosen_color = self.base_colors[action_idx - 52]
            elif 56 <= action_idx <= 59:
                target_idx = 53  # Wild +4
                chosen_color = self.base_colors[action_idx - 56]
            else:
                target_idx = action_idx

            # Find the card in hand
            for card in self.agent_hand:
                if card.to_index() == target_idx:
                    chosen_card = card
                    break

            if chosen_card is None:
                return self._get_state(), -100.0, True, self.get_legal_mask()

            self.agent_hand.remove(chosen_card)
            self.discard_pile_counts[target_idx] += 1
            self.top_card = chosen_card

            # Apply the AI's chosen color!
            if chosen_card.color == CardColor.WILD:
                self.current_color = chosen_color
            else:
                self.current_color = chosen_card.color

        # --- Win / Loss logic remains exactly the same below ---
        if len(self.agent_hand) == 0:
            return self._get_state(), 100.0, True, self.get_legal_mask()

        if np.random.random() > 0.5:
            self.opponent_hand_size -= 1
        else:
            self.opponent_hand_size += 1

        if self.opponent_hand_size <= 0:
            action_cards = sum(
                20.0
                for c in self.agent_hand
                if c.value in [CardValue.SKIP, CardValue.REVERSE, CardValue.DRAW_TWO]
            )
            wild_cards = sum(50.0 for c in self.agent_hand if c.color == CardColor.WILD)
            return (
                self._get_state(),
                -(action_cards + wild_cards) - 10.0,
                True,
                self.get_legal_mask(),
            )

        return self._get_state(), 0.0, False, self.get_legal_mask()
