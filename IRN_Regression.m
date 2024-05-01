% Multiregion Club
% Omar El Sayed, Zachary Loschinskey, Dr. Yujin Han, Dr. Brian Depasquale, Dr. Mike Economo
% April 2024
% Regression of IRN from Behavior

%% Import the data from Yujin's Folder
clear,close all

% add paths for data loading scripts, all fig funcs, and utils
utilspth = "U:\eng_research_economo\YH\Data\Code\Yujin";
addpath(genpath(fullfile(utilspth,'DataLoadingScripts')));
addpath(genpath(fullfile(utilspth,'funcs')));
addpath(genpath(fullfile(utilspth,'utils')));
addpath("U:\eng_research_economo\YH\Data\Code\Yujin\load\");

clc

%% PARAMETERS
params.alignEvent          = 'goCue'; % 'jawOnset' 'goCue'  'moveOnset'  'firstLick'  'lastLick'  'secondLick', 'thirdLick', 'reward'

% time warping only operates on neural data for now.
% TODO: time warp for video and bpod data
params.timeWarp            = 0;  % piecewise linear time warping - each lick duration on each trial gets warped to median lick duration for that lick across trials
params.nLicks              = 15; % number of post go cue licks to calculate median lick duration for and warp individual trials to

params.lowFR               = 1; % remove clusters with firing rates across all trials less than this val
params.persistentUnit      = 0; % YH testing: if 1, only take units that passed persistency test (see removenonPersistentClusters.m)

% set conditions to calculate PSTHs for (and get trial numbers for)
params.condition(1)         = {'hit'};                            
params.condition(end+1)     = {'hit&trialTypes==1'};    % type1=right                                
params.condition(end+1)     = {'hit&trialTypes==2'};    % type2=middle
params.condition(end+1)     = {'hit&trialTypes==3'};    % type3=left                                
params.condition(end+1)     = {'hit&trialTypes==1&Nlicks==2'};                                     
params.condition(end+1)     = {'hit&trialTypes==2&Nlicks==2'};                                     
params.condition(end+1)     = {'hit&trialTypes==3&Nlicks==2'};                                     
params.condition(end+1)     = {'hit&trialTypes==1&Nlicks==5'};                                     
params.condition(end+1)     = {'hit&trialTypes==2&Nlicks==5'};                                     
params.condition(end+1)     = {'hit&trialTypes==3&Nlicks==5'};
params.condition(end+1)     = {'hit&Nlicks==2'};        % Nlicks2=water released aft 2nd lick                                
params.condition(end+1)     = {'hit&Nlicks==5'};        % Nlicks5=water released aft 5th lick                             


% time from align event to grab data for
params.tmin = -1.0;
params.tmax = 2.5;
params.dt = 1/100; % bin size

% smooth with causal gaussian kernel
params.smooth = 20;

% cluster qualities to use
params.quality = {'good'}; % accepts any cell array of strings - special character 'all' returns clusters of any quality

% params.traj_features = {{'tongue','left_tongue','right_tongue','jaw','trident','nose'},...
%     {'top_tongue','topleft_tongue','bottom_tongue','bottomleft_tongue','jaw','top_paw','bottom_paw','top_nostril','bottom_nostril'}};
params.traj_features = {{'tongue','left_tongue','right_tongue','jaw','trident','nose'},...
    {'top_tongue','topleft_tongue','bottom_tongue','bottomleft_tongue','jaw','top_nostril','bottom_nostril'}};

params.feat_varToExplain = 80; % num factors for dim reduction of video features should explain this much variance

params.advance_movement = 0.0; % not sure if still being used anywhere, leaving in place just in case

params.behav_only = 0; % if 1, don't process neural data - speeds up analysis if you only need behavior data

%% SPECIFY DATA TO LOAD

% this path specifies path to a folder structured as
% /data/DataObjects/<MAHXX>/data_structure_XXX.mat
datapth = "U:\eng_research_economo\YH\Data";

meta = [];

% YH16: medulla recording
meta = loadYH16_240306_L_IRN(meta,datapth);
% meta = loadYH16_240306_R_IRN(meta,datapth);
% meta = loadYH10_240307_L_IRN(meta,datapth);
% meta = loadYH10_240307_R_IRN(meta,datapth);
% meta = loadYH10_240308_L_IRN(meta,datapth);
% meta = loadYH10_240308_R_IRN(meta,datapth);

params.probe = {meta.probe}; % put probe numbers into params, one entry for element in meta, just so i don't have to change code i've already written

%% LOAD DATA

% ----------------------------------------------
% -- Neural Data --
% obj (struct array) - one entry per session
% params (struct array) - one entry per session
% ----------------------------------------------
[obj,params] = loadSessionData(meta,params);  % The PSTHs are loaded here
                                              % they look like they are
                                              % already stored in the
                                              % data file (?)

% kinematics
for sessix = 1:numel(meta)
    me(sessix) = loadMotionEnergy(obj(sessix), meta(sessix), params(sessix), datapth);
    kin(sessix) = getKinematics(obj(sessix), me(sessix), params(sessix));
end


%% Prepare Trial Averaged PSTHs and Conduct PCA
clearvars -except meta obj params kin me
cond2use = [2 3 4];  % use right hit, left hit, and center hit 

% 350 time points because 1/100 s bins and -1.0 to 2 secs around go que
psth = obj(1).psth(:,:,cond2use);  % Shape (350, 82, 3) (time, clu, cond)
% Are these already trial averaged?

% Convert (350x3, 82) aka vertically concatenate the conditions
data = [psth(:,:,1); psth(:,:,2); psth(:,:,3)]; % (TimexCond, clu)

% Construct the deviation matrix
D = data - mean(data);

% Calculate the cov matrix S
S = (1 / length(D(:,1)) - 1) * (D'*D);

R = corrcoef(D);

% Calculate eigenvalues and eigenvectors of the cov matrix
[eig_vec, lambda] = eig(S);
eig_vals = diag(lambda);

% Calculate variance explained by each PC
explained_var = eig_vals ./ sum(eig_vals);

% Calculate the cumulative explained variance
cum_explained_var = zeros(1, length(explained_var));
cum_explained_var(1) = explained_var(1);

for chim = 2:length(explained_var)
    cum_explained_var(chim) = cum_explained_var(chim-1) + explained_var(chim);
end

figure()
bar(explained_var)
title("Explained Variance of Trial Averaged Data")
xlabel("Principal Comonent")
ylabel("Explained Variance (%)")
xlim([0 15])
grid on;

% Visualize the cumulative explained variance
figure()
plot(cum_explained_var, "LineWidth",2)
title("Cumulative Explained Variance of Trial Averaged Data")
xlabel("Number of Principal Components")
ylabel("Variance Explained (%)")
grid on;
ylim([min(cum_explained_var)-0.05, max(cum_explained_var)+0.05])


% Select the first num_PCs eigenvectors
PCs = eig_vec(:, 1:3);

% Project the data onto the first num_PCs PCs
projected_data = D * PCs;

% Plot a 3d plot of the first three PCs
figure()
plot3(projected_data(:,1), projected_data(:,2), projected_data(:,3), '.')
xlabel("PC1")
ylabel("PC2")
zlabel("PC3")
title("Trial Average PC1 vs PC2 vs PC3")

% Visualize the first three PCs
figure;
for i = 1:3
    for j = 1:3
        subplot(3, 3, (i-1)*3 + j);
        if i == j
            histogram(projected_data(:, i), 30);
            xlabel(['PC' num2str(i)]);
            ylabel('Frequency');
        else
            scatter(projected_data(:, j), projected_data(:, i), '.');
            xlabel(['PC' num2str(j)]);
            ylabel(['PC' num2str(i)]);
        end
        if i == 1 && j == 1
            title('Histogram of PC1');
        elseif i == j
            title(['Histogram of PC' num2str(i)]);
        else
            title(['Scatter of PC' num2str(j) ' vs PC' num2str(i)]);
        end
    end
end


%% Prepare Single Trial Data and Conduct PCA
trials2use = params(1).trialid(cond2use);
trialdat = obj(1).trialdat; % single trial neural data (time,clu,trials)
trialdat = cellfun(@(x) trialdat(:,:,x), trials2use, 'UniformOutput',false); % get single trial data by cond and index same clusters as above

% Data is now 93100x82 where we have time vs cluster
% The trials are vertically concatenated and they include
% the L, R, and C trials in a row. This matrix is easy to 
% delete rows from when Omar sends the nan indices from the behavior
% features

data = trialdat{1}(:,:,1);

[~,~,k] = size(trialdat{1});
for chim = 2:k
    data = [data; trialdat{1}(:,:,chim)];
end

[~,~,k] = size(trialdat{2});
for chim = 1:k
    data = [data; trialdat{2}(:,:,chim)];
end

[~,~,k] = size(trialdat{3});
for chim = 1:k
    data = [data; trialdat{3}(:,:,chim)];
end

%% PCA on the single trial data
% Construct the deviation matrix
D = data - mean(data);

% Calculate the cov matrix S
S = (1 / length(D(:,1)) - 1) * (D'*D);

% Calculate eigenvalues and eigenvectors of the cov matrix
[eig_vec, lambda] = eig(S);
eig_vals = diag(lambda);

% Calculate variance explained by each PC
explained_var = eig_vals ./ sum(eig_vals);

% Calculate the cumulative explained variance
cum_explained_var = zeros(1, length(explained_var));
cum_explained_var(1) = explained_var(1);

for chim = 2:length(explained_var)
    cum_explained_var(chim) = cum_explained_var(chim-1) + explained_var(chim);
end

figure()
bar(explained_var)
title("Explained Variance of Single Trial Data")
xlabel("Principal Comonent")
ylabel("Explained Variance (%)")
xlim([0 15])
grid on;

% Visualize the cumulative explained variance
figure()
plot(cum_explained_var, "LineWidth",2)
title("Cumulative Explained Variance of Single Trial Data")
xlabel("Number of Principal Components")
ylabel("Variance Explained (%)")
grid on;
ylim([min(cum_explained_var)-0.05, max(cum_explained_var)+0.05])


%% Project data onto first 12 PCs to explain 80% variance
% Number of Principal Components to keep
num_PCs = 12;

% Select the first num_PCs eigenvectors
PCs = eig_vec(:, 1:num_PCs);

% Project the data onto the first num_PCs PCs
projected_data = D * PCs;

% Plot a 3d plot of the first three PCs
figure()
plot3(projected_data(:,1), projected_data(:,2), projected_data(:,3), '.')
xlabel("PC1")
ylabel("PC2")
zlabel("PC3")
title("Single Trial PC1 vs PC2 vs PC3")

% Visualize the first three PCs
figure;
for i = 1:3
    for j = 1:3
        subplot(3, 3, (i-1)*3 + j);
        if i == j
            histogram(projected_data(:, i), 30);
            xlabel(['PC' num2str(i)]);
            ylabel('Frequency');
        else
            scatter(projected_data(:, j), projected_data(:, i), '.');
            xlabel(['PC' num2str(j)]);
            ylabel(['PC' num2str(i)]);
        end
        if i == 1 && j == 1
            title('Histogram of PC1');
        elseif i == j
            title(['Histogram of PC' num2str(i)]);
        else
            title(['Scatter of PC' num2str(j) ' vs PC' num2str(i)]);
        end
    end
end


%% Prepare Kinematics Data - X features 

% Add Features 
feat_labels = {'tongue_angle', 'tongue_length', 'jaw_xdisp_view1', 'jaw_ydisp_view1', 'jaw_xvel_view1', 'jaw_yvel_view1'};

X_features = []; % Initialize X matrix to store all behavioral variables (350xtrials, features)

NaN_Matrix = []; % Initialize NaN matrix -- NaN: 0. (350xtrials, features)

for feat_no = 1: length(feat_labels)

    % Select Feature 
    feature = feat_labels(feat_no);

    % Find Feature index in kin struct
    featix = find(ismember(kin(1).featLeg,feature));
    
    X_feat = [];
    X_NaN = [];

    for trial_type = 1:3
        % Get Data for each trial type 
        trial_type_data = kin(1).dat(:,params(1).trialid{1+trial_type}, featix);
        X_temp = reshape(trial_type_data, [], 1);
        NaN_indices = isnan(X_temp); % NaN entries = 1 
        X_feat = (vertcat(X_feat, X_temp));  % (350xtrials, 1)
        X_NaN = (vertcat(X_NaN, NaN_indices)); % (350xtrials, 1)
    end
    % Stores a Feature each Iteration
    X_features = (horzcat(X_features, X_feat)); % (350xtrials, features)
    NaN_Matrix = (horzcat(NaN_Matrix, X_NaN));  % (350xtrials, features)
end

X = X_features(~any(NaN_Matrix, 2), :); 


%% Cross Validation to Optimize # of PCs to Use.

PCs_vec = 1:1:50; 

% Initialize Matrix to store CV performance for each cumulative PC
PC_PERF = [];

for PC_iter = 1:length(PCs_vec)
    PC = PCs_vec(PC_iter); 
    R_squared = [];

    
    
    for fold = 1:10
        kf = cvpartition(size(X,1), 'HoldOut', 0.2);

        train_idx = training(kf, 1);
        test_idx = test(kf, 1);
        
        % Split X into Training and Testing Sets
        X_train = X(train_idx, :);
        X_test = X(test_idx, :);

        % Select the first num_PCs eigenvectors
        PCs = eig_vec(:, 1:PC);
        
        % Project the data onto the first num_PCs PCs
        projected_data = D * PCs;
        Real_Activity = projected_data(~any(NaN_Matrix, 2), :);
        
        Y_train = Real_Activity(train_idx, :);
        Y_test = Real_Activity(test_idx, :);
        
        % Solution
        B_OLS = (X_train' * X_train) \ (X_train' * Y_train);

        % Test Prediction
        Y_pred = X_test * B_OLS; 

        % Performance (R^squared)
        r_squared = 1 - (sum((Y_test - Y_pred).^2) / sum((Y_test - mean(Y_test)).^2));
        R_squared = ([R_squared, r_squared]);
    end
    pc_perf = mean(R_squared);
    PC_PERF = ([PC_PERF; pc_perf]);
end

%% Plot CV PERFORMANCE OF Optimal PC to use

% Determine PC to use 
[max_perf, PC_idx] = max(PC_PERF);

figure();
plot(PCs_vec', PC_PERF,'-', 'color', 'b', 'linewidth', 3);
hold on
xline(PCs_vec(PC_idx), '--','k')
xlabel('Principal Component')
ylabel('R^{2}');
title('CV Results: Prediction Performance', 'Behavior to Varying PCs of IRN');
legend(['PC = ' num2str(PCs_vec(PC_idx)) '; R^{2} = ' num2str(max(PC_PERF))], 'fontsize', 12)
ylim([0.1 0.7]);

%% Cross Validation to Determine Best Lambda

% Project Data onto Optimal PCs
% Select the first num_PCs eigenvectors
PCs = eig_vec(:, 1:PCs_vec(PC_idx));

% Project the data onto the first num_PCs PCs
projected_data = D * PCs;
Real_Activity = projected_data(~any(NaN_Matrix, 2), :);

% Create Lambda Values
lambda_values = logspace(-3,3,200);

% Initialize Matrix to store CV performance for each Lambda Value
RIDGE_PERF = [];

for ridge_iter = 1:length(lambda_values)
    lmbda = lambda_values(ridge_iter); 
    R_squared = [];

    
    for fold = 1:10
        kf = cvpartition(size(X,1), 'HoldOut', 0.2);
        train_idx = training(kf, 1);
        test_idx = test(kf, 1);
        
        % Split X into Training and Testing Sets
        X_train = X(train_idx, :);
        X_test = X(test_idx, :);
        
        Y_train = Real_Activity(train_idx, :);
        Y_test = Real_Activity(test_idx, :);
        
        % Solution
        B_Ridge = (X_train' * X_train + lmbda*eye(size(X_train,2))) \ (X_train' * Y_train);

        % Test Prediction
        Y_pred = X_test * B_Ridge; 

        % Performance (R^squared)
        r_squared = 1 - (sum((Y_test - Y_pred).^2) / sum((Y_test - mean(Y_test)).^2));
        R_squared = ([R_squared, r_squared]);
    end
    ridge_perf = mean(R_squared);
    RIDGE_PERF = ([RIDGE_PERF; ridge_perf]);
end

%% Plot CV PERFORMANCE OF Optimal Lambda to use

% Determine Optimal Lambda to use 
[max_perf, lmbda_idx] = max(RIDGE_PERF);

figure();
semilogx(lambda_values', RIDGE_PERF,'-', 'color', 'b', 'linewidth', 3);
xlabel('\lambda')
ylabel('R^{2}');
title('CV Results: Prediction Performance of Different \lambda', 'Behavior to 4 PCs to IRN');
legend(['\lambda = ' num2str(lambda_values(lmbda_idx)) '; R^{2} = ' num2str(max(RIDGE_PERF))], 'fontsize', 12)
ylim([0.1 0.7]);