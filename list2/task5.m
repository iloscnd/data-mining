N = 10000;

prize = randi(3, [N,1]);
first_choice = randi(3, [N,1]);

change_strat = prize ~= first_choice;
stay_strat = prize == first_choice;

fprintf("Stay stats %.5d:\n", sum(stay_strat)/N)
fprintf("Change stats %.5d:\n", sum(change_strat)/N)