function W = generate_JointMatrix(A,B,n_perc)

fc_mat_1 = corr(A');
fc_mat_2 = corr(B');

% keep only positive values
fc_mat_1 = fc_mat_1 .* (fc_mat_1 > 0);
fc_mat_2 = fc_mat_2 .* (fc_mat_2 > 0);

fprintf('- top percent\n')
fc_mat_1 = fc_mat_1 .* (fc_mat_1 > repmat(prctile(fc_mat_1,n_perc,2),1,size(fc_mat_1,2)));
fc_mat_1(fc_mat_1<0) = 0;

fprintf('- top percent\n')
fc_mat_2 = fc_mat_2 .* (fc_mat_2 > repmat(prctile(fc_mat_2,n_perc,2),1,size(fc_mat_2,2)));
fc_mat_2(fc_mat_2<0) = 0;

fprintf('- kernel \n')
G_1 = normcorr(fc_mat_1', fc_mat_1');
G_2 = normcorr(fc_mat_2', fc_mat_2');

fprintf('- compare fc to landmarks across individuals \n')
G_12 = normcorr(fc_mat_1', fc_mat_2');

fprintf('- stitch it together \n')
W = [ G_1 G_12 ; G_12' G_2 ];
W = sparse(double(W));


end


function W = normcorr(A,B)

A1 = sqrt(sum(A .* A, 1)); A1(A1 == 0) = 1;
B1 = sqrt(sum(B .* B, 1)); B1(B1 == 0) = 1;
W = bsxfun(@times, A' * B, 1 ./ A1');
W = bsxfun(@times, W, 1 ./ B1);

end
