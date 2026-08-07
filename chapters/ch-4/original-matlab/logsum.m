function result = logsum(logv)

%    |result = logsum(logv)| returns the log sum of elements of vector
%    |logv|. Operations are performed to maximize accuracy.
%
%    Note that:
%    logsum([log(a) log(b)]) = log(a+b);
%    logsum([log(a) Inf])    = Inf;
%    logsum([log(a) -Inf])   = log(a);
%    logsum([Inf Inf])       = Inf;
%    logsum([-Inf Inf])      = Inf;
%    logsum([Inf -Inf])      = Inf;
%    logsum([-Inf -Inf])     = -Inf;


len = length(logv);
if (len == 1) 
    % Single term
    result = logv;
else
    % First two terms
    if (logv(1) == Inf) & (logv(2) == Inf)
        result = Inf;
    elseif (logv(1) == Inf) & (logv(2) == -Inf)
        result = Inf;
    elseif (logv(1) == -Inf) & (logv(2) == Inf)
        result = Inf;
    elseif (logv(1) == -Inf) & (logv(2) == -Inf)
        result = -Inf;
    elseif (logv(1) > logv(2))
        result = logv(1) + log( 1 + exp( logv(2)-logv(1) ) );
    else
        result = logv(2) + log( 1 + exp( logv(1)-logv(2) ) );
    end
    % Remaining terms
    for (i=3:len)
        term = logv(i);
        if (result == Inf) & (term == Inf)
            result = Inf;
        elseif (result == Inf) & (term == -Inf)
            result = Inf;
        elseif (result == -Inf) & (term == Inf)
            result = Inf;
        elseif (result == -Inf) & (term == -Inf)
            result = -Inf;
        elseif (result > term)
            result = result + log( 1 + exp( term-result ) );
        else
            result = term   + log( 1 + exp( result-term ) );
        end
    end
end
