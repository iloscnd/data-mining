

x = bitxor(3, bitxor(2,1));
assert(x == 0);
prize = randi(3, [10000,1]);
first_choice = randi(3, [10000,1]);
second_choice = bitxor(prize, first_choice);
