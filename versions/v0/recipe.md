1. perception + value
loss: GTO EV
input: random situation from GTO simulation (env - https://github.com/zer0o0ne/PokerMegabot_4000, GTO - https://github.com/sol5000/gto), approximately 1M (5 days) simulations

2. action
loss: cross entropy on best solution
input: MCTS tree with value labels

3. perception + action + value
loss: MCTS-based action + value evolution
input: all-time game history (including current situation)