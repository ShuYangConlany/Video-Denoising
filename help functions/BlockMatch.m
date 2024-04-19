function [ motionCompen] = BlockMatch(Ik, Il, BlockSize, D)
    [m, n] = size(Ik);
%     motionErr = zeros(m, n);
    motionCompen = zeros(m,n);
%     motions = zeros(floor(m/BlockSize), floor(n/BlockSize), 2);
    for i = 1 : floor(m / BlockSize)
        for j = 1 : floor(n / BlockSize)
            Block_cur = Ik((i-1)*BlockSize + 1 : i * BlockSize, (j-1)*BlockSize + 1 : j * BlockSize);
            minError = inf;
            row_head = (i-1)*BlockSize + 1;
            col_head = (j-1)*BlockSize + 1;          
            for d1 = -D : D
                for d2 = -D : D
                    di = row_head + d1;
                    dj = col_head + d2;
                    if di > 0 && dj > 0 && di + BlockSize - 1 <= m && dj + BlockSize - 1 <= n
                        Block_matched = Il(di : di + BlockSize - 1, dj : dj + BlockSize - 1);
                        Error = mean(mean(abs(Block_cur - Block_matched)));
                        if Error <= minError
                            minError = Error;
                            d1_best = d1;
                            d2_best = d2;
                        end
                    end 
                end
            end
%             motions(i, j, 1) = d1_best;
%             motions(i, j, 2) = d2_best;
            for bi = 1:BlockSize
                for bj = 1:BlockSize
                    row_idx = row_head + bi - 1;
                    col_idx = col_head + bj - 1;
                    moved_row_idx = row_idx - d1_best; 
                    moved_col_idx = col_idx - d2_best; 
                    if moved_row_idx > 0 && moved_col_idx > 0 && moved_row_idx <= m && moved_col_idx <= n
%                         motionErr(row_idx, col_idx) = Il(row_idx, col_idx) - Ik(moved_row_idx, moved_col_idx);
                        motionCompen(row_idx, col_idx) = Ik(moved_row_idx, moved_col_idx);
                    end
                end
            end
        end
    end
end