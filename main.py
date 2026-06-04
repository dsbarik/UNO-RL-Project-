from rich import print

from environment import Deck

deck = Deck()

for i in range(10):
    print(deck.draw())
    print()
