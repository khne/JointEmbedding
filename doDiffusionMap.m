function embed = doDiffusionMap(W,n_template,n_components)

fprintf('- diffusion map embedding\n')

alpha = 0.5;
diffusion_time = 1;

n = size(W,1);

fprintf('- normalize\n')
d_alpha     = spdiags(sum(W,2).^-alpha, 0, speye(n));
L_alpha     = d_alpha * W * d_alpha;

fprintf('- laplacian\n')
d_alpha     = spdiags(sum(L_alpha,2).^-1, 0, speye(n));
M           = d_alpha * L_alpha;

fprintf('- eigendecomposition\n')
[V,E]       = eigs(M, n_components + 1);
[lambdas, sorted_indices] = sort(diag(E),'descend');
vectors     = V(:, sorted_indices);

fprintf('- diffusion map\n')
psi = vectors./repmat(vectors(:,1),1,size(vectors,2));

lambdas(2:end) = lambdas(2:end).^diffusion_time;

embedding = psi(:, 2:(n_components + 1)) * spdiags(lambdas(2:n_components+1), 0, n_components, n_components);
lambdas = lambdas(2:end);

embedding = embedding;

%% ========================================================================
fprintf('- results\n')

embed = [];
embed.lambdas   = lambdas;

if n_template == 0    
    embed.individual       = embedding;
else
    embed.template         = embedding(1:n_template,:);
    embed.individual       = embedding(n_template+1:end,:);
end

%% ========================================================================


