function W = generate_SingleMatrix(A,n_perc)

fc_mat = A;

% keep only positive values
fc_mat = fc_mat .* (fc_mat > 0);

fprintf('- top percent\n')
fc_mat = fc_mat .* (fc_mat > repmat(prctile(fc_mat,n_perc,2),1,size(fc_mat,2)));
fc_mat(fc_mat<0) = 0;

fprintf('- kernel \n')
W = normcorr(fc_mat', fc_mat');
W = sparse(double(W));


end


function W = normcorr(A,B)

A1 = sqrt(sum(A .* A, 1)); A1(A1 == 0) = 1;
B1 = sqrt(sum(B .* B, 1)); B1(B1 == 0) = 1;
W = bsxfun(@times, A' * B, 1 ./ A1');
W = bsxfun(@times, W, 1 ./ B1);

end
