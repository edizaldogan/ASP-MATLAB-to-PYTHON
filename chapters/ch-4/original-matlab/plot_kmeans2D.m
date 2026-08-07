function [] = plot_kmeans2D(data,means)

%   |plot_kmeans2D(data,means)| plots the clusters associated 
%   with |means| in |data| samples, using a Euclidian distance. 
%   |means| is a cell array containing the prototype vectors.

[numPts,dim] = size(data);
numClust = length(means);
  
% Make colormap for classification
cmap = hsv(numClust);
  
% Compute clusters  
for i=1:numClust,
  x = data - repmat(means{i},numPts,1);
  dist(i,:) = sum( (x .* x)' );
end;
[minDist,whereMin] = min(dist);
  
cla;
circle = [cos(linspace(-pi, pi, 100)') sin(linspace(-pi, pi, 100)')];
for i=1:numClust,
       subData = data(whereMin==i,:);
       line( subData(:,1), subData(:,2), ...
	     'linestyle','none','marker','.','color',cmap(i,:));
       line(means{i}(1),means{i}(2), 10, ...
         'marker','o','markersize',12,'color',[1 1 1],'linew',2);
end;