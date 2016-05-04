function test_pose_avp(filename)
    cls_names = { 'aeroplane','bicycle','boat','bottle','bus','car','chair','diningtable','motorbike','sofa','train','tvmonitor'};
    dets_from_file = importdata(filename);
    % keep  only azimuth
    dets_from_file = [dets_from_file(:,1:7) dets_from_file(:,9)];
    addpath('/media/dey/debidatd/pascal3d/VDPM/');
    
    N = max(dets_from_file(:,1))+1;
    
    for vnum = [4, 8, 12, 16]
        dets_all = cell(N, 1);
        for p = 1:12
            for i = 1:N
                azimuth_interval = [0 (360/(vnum*2)):(360/vnum):360-(360/(vnum*2))];
                dets_all_tmp = dets_from_file(dets_from_file(:,1)==(i-1) & dets_from_file(:,2)==p, 3:8);
                for j = 1:size(dets_all_tmp, 1)
                    dets_all_tmp(j, 5) = find_interval(dets_all_tmp(j, 5), azimuth_interval);
                end
                dets_all{i,1} = dets_all_tmp;%combine_bbox_azimuthview(dets_all_tmp, azimuth_interval); 
            end
    
            [recall, precision, accuracy, ap, aa] = compute_recall_precision_accuracy(cls_names{p}, dets_all, vnum);
            end
        end
   end
function ind = find_interval(azimuth, a)

for i = 1:numel(a)
    if azimuth < a(i)
        break;
    end
end
ind = i - 1;
if azimuth > a(end)
    ind = 1;
end
end
