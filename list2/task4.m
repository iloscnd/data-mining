permutations = perms(1:9);

%good = 0;
%while good == 0
%    board = permutations(randperm(factorial(9),9), :)';
%    good = 1;
%    if numel(unique(board, 'rows')) ~= 81
%        good = 0;
%        continue
%    end
%    
%    for i = 1:3:7
%        for j = 1:4:7
%            if numel(unique(board(i:(i+2), j:(j+2)))) ~= 9
%                disp(board);
%                disp(i);
%                disp(j);
%                disp(board(i:(i+2), j:(j+2)))
%                good = 0;
%           end
%            if good == 0
%                break
%            end
%        end
%        if good == 0
%            break
%        
%    end
%end
%disp(board)