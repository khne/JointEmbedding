

load('data_demo.mat','REF_FC','FC_A','FC_B','ndata')


n_perc = 90;
n_grads = 3;

% average connectivity to span reference space
% map from stored triu vector back to matrix
TEMPLATE = zeros(ndata);
TEMPLATE(triu(true(ndata),1)) = REF_FC;
TEMPLATE = TEMPLATE + TEMPLATE' + eye(ndata);
n_template = size(TEMPLATE,1);

DATA_A = zeros(ndata);
DATA_A(triu(true(ndata),1)) = FC_A;
DATA_A = DATA_A + DATA_A' + eye(ndata);

DATA_B = zeros(ndata);
DATA_B(triu(true(ndata),1)) = FC_B;
DATA_B = DATA_B + DATA_B' + eye(ndata);

% -------------------------------------------------------------------------

fprintf('create embedding of reference\n')
W_template = generate_SingleMatrix(TEMPLATE,n_perc);
embed_template = doDiffusionMap(W_template,n_template,n_grads);

fprintf('joint embedding between reference and indiviuduals\n')
W_TA = generate_JointMatrix(TEMPLATE,DATA_A,n_perc);
embed_TA = doDiffusionMap(W_TA,n_template,n_grads);
W_TB = generate_JointMatrix(TEMPLATE,DATA_B,n_perc);
embed_TB = doDiffusionMap(W_TB,n_template,n_grads);

fprintf('indiviudual embeddings\n')
W_A = generate_SingleMatrix(DATA_A,n_perc);
embed_A = doDiffusionMap(W_A,0,n_grads);
W_B = generate_SingleMatrix(DATA_B,n_perc);
embed_B = doDiffusionMap(W_B,0,n_grads);

% -------------------------------------------------------------------------

fprintf('align embeddings to reference space\n')
X = embed_template.template;

fprintf('align joint embeddings via the group component\n')
Y = embed_TA.template;
[d,Z,tr] = procrustes(X,Y,'scaling',false);
embed_TA.template_to_template = Z;
embed_TA.individual_to_template = tr.b * embed_TA.individual * tr.T + repmat(tr.c(1,:),size(Y,1),1);

Y = embed_TB.template;
[d,Z,tr] = procrustes(X,Y,'scaling',false);
embed_TB.template_to_template = Z;
embed_TB.individual_to_template = tr.b * embed_TB.individual * tr.T + repmat(tr.c(1,:),size(Y,1),1);

fprintf('align individual embeddings directly\n')
Y = embed_A.individual;
[d,Z,tr] = procrustes(X,Y,'scaling',false);
embed_A.individual_to_template = Z;

Y = embed_B.individual;
[d,Z,tr] = procrustes(X,Y,'scaling',false);
embed_B.individual_to_template = Z;

% -------------------------------------------------------------------------
fprintf('- visualize\n')

cm = lines(4);

fig = figure('Visible','on','Units','Normalized','Position',[.1 .1 .6 .4]);
subplot(1,2,1); hold on;
plot(embed_template.template(:,1),embed_template.template(:,2),'k.')
plot(embed_TA.individual_to_template(:,1),embed_TA.individual_to_template(:,2),'.')
plot(embed_TB.individual_to_template(:,1),embed_TB.individual_to_template(:,2),'.')
xlabel('Component 1'); ylabel('Component 2'); zlabel('Component 3');
legend({'REFERENCE','DATA-A','DATA-B'},'location','best')
legend boxoff
title('JOINT EMBEDDING')

subplot(1,2,2); hold on;
plot(embed_template.template(:,1),embed_template.template(:,2),'k.')
plot(embed_A.individual_to_template(:,1),embed_A.individual_to_template(:,2),'.')
plot(embed_B.individual_to_template(:,1),embed_B.individual_to_template(:,2),'.')
xlabel('Component 1'); ylabel('Component 2'); zlabel('Component 3');
legend({'REFERENCE','DATA-A','DATA-B'},'location','best')
legend boxoff
title('INDIVIDUAL EMBEDDING')


% -------------------------------------------------------------------------
return;


