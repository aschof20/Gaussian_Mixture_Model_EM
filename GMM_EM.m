% Implementation of the Expectation-Maximisation (EM) Guassian Mixture
% Model.

% Import the dataset.
imported_data = importdata("old_faithful.txt");
data = imported_data.data;
 
%%% Initialisation of variables. %%%
K = 2;                  % Number of clusters.
N = length(data);       % Sample size.
D = size(data,2)-1;     % Dimensions of the dataset.
pi = [0.5 0.5];         % Mixing coefficients.

% Initialise Diagonal Covariance Matrices for each cluster.
Sigma = zeros(D,D,K);   
for k=1:K
   Sigma(:,:,k) = eye(D,D); 
end
 
% Randomly initialise cluster means.
means = [(max(data(:,2))-min(data(:,2)))*rand(K,1) +...
 min(data(:,2)) (max(data(:,3))-min(data(:,3)))*rand(K,1) + min(data(:,3))];
% Empty matrix to store means from previous iteration.
old_means = zeros(K, D); 

%%% EM Algorithm %%%

% Continue to iterate and update means while the difference between one
% iteration to the next is greater than 1e^-2.
while (sum(sum(abs(means - old_means))) > 1e-2)
    plotSoftKMeans(data, means, Sigma)
    % Zeros matrix to store cluster responsibilites to data points.
    responsibility = zeros(N,K);
    
    % Update the old_means matrix with the means of the current iteration.
    old_means = means;
  
    % E STEP:
    % Loop over the data points and the clusters to calculate cluster
    % responsibilities for the respective data points.
    for n = 1:N
        for k = 1:K
           responsibility(n,k) = pi(k)*gaussPdf(data(n, 2:3),means(k,:),Sigma(:,:,k));
        end
        % Normalise the responsibilies for the respective clusters. 
        responsibility(n,:) = responsibility(n,:)./sum(responsibility(n,:));
    end

    
    % M STEP:
    % Loop over the clusters and data dimensions to update the cluster
    % means based on the responsibility of a cluster to a data point.
    for k = 1:K
        for d = 1:D
            means(k, d) = sum(responsibility(:,k).*data(:,d+1));
        end
        % Update the means, mixing coefficients
        means(k,:) = means(k,:)/sum(responsibility(:,k));
        pi(k)= sum(responsibility(:,k))/N;
    end
    
    % Update covariance matrices for each cluster.
    % Loop over the each data point in each dimension and calculate the
    % covariance matrices for each cluster.
    for i = 1:N
        for d =1:D
            % Calculate the difference between the data point and the
            % respective means.
            Xm = bsxfun(@minus, data(:,2:3), means(d,:));
            
            % Responsibitilies times the differences between data points
            % and cluster means.
            z=bsxfun(@times, Xm, responsibility(:,d));
 
            % Calculate the covariance matrix for each cluster.
            Sigma(:,:,d) = (z)'*((Xm))./sum(responsibility(:,d));
        end
    end
end
%%%% end of algorithm %%%%

% Function to calculate the probability density function.
function gaussPdf = gaussPdf(X, means, sigma)

    % Calculate a vector of the data point residuals.
    Xmu = X-means;
    
    % Calculate the probability density function with the residuals and
    % diagonal covariance matrix.
    gaussPdf = 1/sqrt(det(sigma)*(2*pi)^2) * exp(-0.5*diag(Xmu*inv(sigma)*Xmu'));
end

% Function to plot the soft k-means centroids and contour plot overlays
function plotSoftKMeans = plotSoftKMeans(data, means, Sigma)
    figure(1); clf
    
    % Scatter plot of the data.
    scatter(data(:,2), data(:,3)); axis square; box on
    axis([min(data(:,2))-0.5 max(data(:,2))+0.5 min(data(:,3))-0.5,max(data(:,3))+0.5])
    hold on
    
    % Cluster centroids (means).
    plot(means(:,1), means(:,2), "rx" , "markersize", 10)
    hold on
    
    % Overlay contour plot to the scatterplot.
    y1 = linspace(min(data(:,2))-0.5,max(data(:,2))+0.5);
    y2 = linspace(min(data(:,3))-0.5,max(data(:,3))+0.5);
    [X1 X2] = meshgrid(y1, y2);
    
    % Loop over the means and generate contour plots for the respective
    % clusters.
    for i = 1:length(means)
        % Probability density functions for the respective clusters.
        gPDF = gaussPdf([X1(:) X2(:)], reshape(means(i,:),1,2), Sigma(:,:,i));
        % Contour plots of the respective clusters.
        contour(X1, X2, reshape(gPDF, size(X2)),5);
    end
    
    % Other plot elements.
    xlabel("eruption duration (min)"); 
    ylabel("time to next eruption (min)"); 
    colorbar;
    
    hold off
end