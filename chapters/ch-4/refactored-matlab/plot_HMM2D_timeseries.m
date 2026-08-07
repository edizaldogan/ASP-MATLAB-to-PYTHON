function [hy,h] = plot_HMM2D_timeseries(x,stateSeq)

%   |plot_HMM2D_timeseries(x,statesSeq)| plots a two-dimensional sequence |x|
%   (one observation per row) as two separate figures, one per dimension. 
%   It superposes the corresponding state sequence |stateSeq| as colored dots 
%   on the obesrvations. |x| and |stateSeq| must have the same length.

len = size(x,1);
numStates = max(stateSeq);

cmap = hsv(numStates);

plot(x(:,1));
hold on;
plot(x(:,2));

for i=1:numStates,
  [where] = find(stateSeq ~= i);
  copy = x;
  copy(where,:) = NaN;

  plot(copy(:,1),'color',cmap(i,:), ...
      'marker','o','markerface',cmap(i,:),'linestyle','none');
 
  plot(copy(:,2),'color',cmap(i,:), ...
      'marker','o','markerface',cmap(i,:),'linestyle','none');
end;
hold off;
