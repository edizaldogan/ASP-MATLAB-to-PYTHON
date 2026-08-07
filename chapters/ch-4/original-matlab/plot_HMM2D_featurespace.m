function [hy,h] = plot_HMM2D_featurespace(x,stateSeq)

%    |plot_HMM2D_featurespace(x,stateSeq)| plots a two-dimensional sequence
%    |x| (one observation per row) as a 2-dimensional figure. It superposes
%    the corresponding state sequence |stateSeq| as colored dots on the
%    obesrvations. |x| and |stateSeq| must have the same length.

len = size(x,1);
numStates = max(stateSeq);

cmap = hsv(numStates);

hy(1) = plot(x(:,1),x(:,2));
set(gca,'dataAspectRatio',[1 1 1]);
hold on;

for i=1:numStates,
  [where] = find(stateSeq ~= i);
  copy = x;
  copy(where,:) = NaN;
    
  h(i,1) = plot(copy(:,1),copy(:,2),'color',cmap(i,:), ...
      'marker','o','markerface',cmap(i,:), 'markerSize', 5, ...
      'linestyle','none');
  
  leg{i} = ['State ' num2str(i)];

end;

hold off

legend(h,leg);