function predicted = MotionCompensate(predicted,observed)
    BlockSize=4;
    D=8;
    [predicted]=BlockMatch(predicted, observed, BlockSize, D);
end


