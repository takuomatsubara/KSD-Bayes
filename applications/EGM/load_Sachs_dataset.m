function [X,names] = load_Sachs_dataset()

% import the flow cytometry datasets
disp('Importing dataset...')
filelist = dir('Sachs_dataset');
name = {filelist.name};
X = zeros(0,11);
for i = [3,9:16] % concatenate over th 9 different stimuli datasets
    X = [X; xlsread(['Sachs_dataset/',name{i}])];
end
disp('...done.')

% protein names
names = {'praf','pmek','plcg','PIP2','PIP3','p44/42','pakts473','PKA','PKC','P38','pjnk'}';

% square root transform
X = sqrt(X);

% standardise
X = bsxfun(@rdivide,X,std(X,0,1));

% dataset with outliers removed
X_max = 10; % outlier threshold
outliers = (max(X,[],2) > X_max);
X(outliers,:) = [];

% re-standardise with outliers removed
X = bsxfun(@rdivide,X,std(X,0,1));

% transpose
X = X';

% save
save('X.mat','X','names')

end