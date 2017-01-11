function nms_flag = nms( bbox, score, method, threshould )
    [~, sort_ind] = sort(score, 'descend');
    bbox_num = size(bbox,2);
    nms_flag = true(bbox_num, 1);
    if(strcmp(method, 'overlap'))
        for i = 1 : bbox_num
            for j = i + 1 : bbox_num
                ii = sort_ind(i);
                jj = sort_ind(j);
                x_arr = bbox(1 : 2 : 8, jj);
                y_arr = bbox(2 : 2 : 8, jj);

                x_union = [bbox(1 : 2 : 8, ii); bbox(1 : 2 : 8, jj)];
                y_union = [bbox(2 : 2 : 8, ii); bbox(2 : 2 : 8, jj)];
                union_poly_ind = convhull(x_union, y_union);
                union_area = polyarea(x_union(union_poly_ind), y_union(union_poly_ind));
                insect_area = polygon_intersect(bbox(1 : 2 : 8, ii)',bbox(2 : 2 : 8, ii)', ...
                    bbox(1 : 2 : 8,jj)', bbox(2 : 2 : 8,jj)');
                if(insect_area / union_area > threshould)
                    nms_flag(jj) = false;
                end
            end
        end
    else if(strcmp(method, 'overlap_refine'))
        for i = 1 : bbox_num
            for j = i + 1 : bbox_num
                ii = sort_ind(i);
                jj = sort_ind(j);
                x_arr = bbox(1 : 2 : 8, jj);
                y_arr = bbox(2 : 2 : 8, jj);

                x_union = [bbox(1 : 2 : 8, ii); bbox(1 : 2 : 8, jj)];
                y_union = [bbox(2 : 2 : 8, ii); bbox(2 : 2 : 8, jj)];
                union_poly_ind = convhull(x_union, y_union);
                union_area = polyarea(x_union(union_poly_ind), y_union(union_poly_ind));
                insect_area = polygon_intersect(bbox(1 : 2 : 8, ii)',bbox(2 : 2 : 8, ii)', ...
                    bbox(1 : 2 : 8,jj)', bbox(2 : 2 : 8,jj)');
                area_ii = polyarea(bbox(1 : 2 : 8, ii), bbox(2 : 2 : 8, ii));
                area_jj = polyarea(bbox(1 : 2 : 8, jj), bbox(2 : 2 : 8, jj));
                if(insect_area / union_area > threshould)
                    if nms_flag(ii)==true&&nms_flag(jj)==true
                        if area_ii>area_jj&&(score(ii)+0.3)>score(jj)
                            nms_flag(jj) = false;
                        elseif score(ii)<score(jj)
                            nms_flag(ii) = false;
                        end
                    end
                end
            end
        end
    end
end
% function nms_flag = nms( bbox, score, method, threshould )
%     [~, sort_ind] = sort(score, 'descend');
%     bbox_num = size(bbox,2);
%     nms_flag = true(bbox_num, 1);
%     if(strcmp(method, 'overlap'))
%         for i = 1 : bbox_num
%             ii = sort_ind(i);
%             if nms_flag(ii)==false
%                 continue;
%             end
%             for j = 1 : bbox_num
%                 jj = sort_ind(j);
%                 if j==i
%                     continue;
%                 end
%                 if nms_flag(jj)==false
%                     continue;
%                 end        
%                 x_arr = bbox(1 : 2 : 8, jj);
%                 y_arr = bbox(2 : 2 : 8, jj);

%                 x_union = [bbox(1 : 2 : 8, ii); bbox(1 : 2 : 8, jj)];
%                 y_union = [bbox(2 : 2 : 8, ii); bbox(2 : 2 : 8, jj)];
%                 box_ii=[bbox(1,ii),bbox(2,ii),bbox(5,ii),bbox(6,ii)];
%                 box_jj=[bbox(1,jj),bbox(2,jj),bbox(5,jj),bbox(6,jj)];
%                 union_poly_ind = convhull(x_union, y_union);
%                 union_area = polyarea(x_union(union_poly_ind), y_union(union_poly_ind));
%                 insect_area = polygon_intersect(bbox(1 : 2 : 8, ii)',bbox(2 : 2 : 8, ii)', ...
%                     bbox(1 : 2 : 8,jj)', bbox(2 : 2 : 8,jj)');
%                 if(insect_area / union_area > threshould)
%                     if score(ii)>score(jj)
%                         nms_flag(jj) = false;
%                     elseif (bbox(5,ii)-bbox(1,ii))*(bbox(6,ii)-bbox(2,ii))>(bbox(5,jj)-bbox(1,jj))*(bbox(6,jj)-bbox(2,jj))
%                         nms_flag(jj) = false;
%                     else
%                         nms_flag(ii) = false;
%                         break;
%                     end                       
%                 end
%                 % height diff is less than a quarter of the higher height
%                 if abs((box_ii(4)-box_ii(2))-(box_jj(4)-box_jj(2)))<((box_ii(4)-box_ii(2))+(box_jj(4)-box_jj(2)))/2
%                     % make sure in the same line
%                     % if (abs(box_ii(4)-box_jj(4))<max((box_ii(4)-box_ii(2)),(box_jj(4)-box_jj(2))))/4 && (abs(box_ii(2)-box_jj(2))<max((box_ii(4)-box_ii(2)),(box_jj(4)-box_jj(2))))/4
%                     if abs(box_ii(4)-box_jj(4))+abs(box_ii(2)-box_jj(2))<(max(box_ii(4),box_jj(4))-min(box_ii(2),box_jj(2)))/3 
%                         % make sure in the same location
%                         if (box_ii(1)<=box_jj(1)) && (box_ii(3)+min((box_ii(4)-box_ii(2)),(box_jj(4)-box_jj(2)))>=box_jj(3))
%                             nms_flag(jj)=false;
%                         end
%                     end
%                 end
%             end
%         end
%     else if(strcmp(method, 'overlap_refine'))
%         for i = 1 : bbox_num
%             ii = sort_ind(i);
%             if nms_flag(ii)==false
%                 continue;
%             end
%             for j = 1 : bbox_num
%                 if j==i
%                     continue;
%                 end         
%                 jj = sort_ind(j);
%                 x_arr = bbox(1 : 2 : 8, jj);
%                 y_arr = bbox(2 : 2 : 8, jj);

%                 x_union = [bbox(1 : 2 : 8, ii); bbox(1 : 2 : 8, jj)];
%                 y_union = [bbox(2 : 2 : 8, ii); bbox(2 : 2 : 8, jj)];
%                 union_poly_ind = convhull(x_union, y_union);
%                 union_area = polyarea(x_union(union_poly_ind), y_union(union_poly_ind));
%                 insect_area = polygon_intersect(bbox(1 : 2 : 8, ii)',bbox(2 : 2 : 8, ii)', ...
%                     bbox(1 : 2 : 8,jj)', bbox(2 : 2 : 8,jj)');
%                 area_ii = polyarea(bbox(1 : 2 : 8, ii), bbox(2 : 2 : 8, ii));
%                 area_jj = polyarea(bbox(1 : 2 : 8, jj), bbox(2 : 2 : 8, jj));
%                 ratio=insect_area / union_area;
%                 if(ratio > threshould)
%                     nms_flag(jj) = false;
%                 end
%                 if (bbox(5,ii)-bbox(1,ii))*(bbox(6,ii)-bbox(2,ii))>(bbox(5,jj)-bbox(1,jj))*(bbox(6,jj)-bbox(2,jj)) && ratio>threshould
%                     nms_flag(jj) = false;
%                 end
%                 % link box of the same line
%                 % if abs(bbox(6,ii)-bbox(6,jj))+abs(bbox(2,ii)-bbox(2,jj)))<0.5*min(bbox(6,ii)-bbox(2,ii),bbox(6,jj)-bbox(2,jj))
                    
%             end
%         end
%     end
end
%     else if(strcmp(method, 'cover'))
%              for i = 1 : bbox_num
%                 for j = i + 1 : bbox_num
%                     ii = sort_ind(i);
%                     jj = sort_ind(j);

%                     x_union = [bbox(1 : 2 : 8, ii); bbox(1 : 2 : 8, jj)];
%                     y_union = [bbox(2 : 2 : 8, ii); bbox(2 : 2 : 8, jj)];
%                     union_poly_ind = convhull(x_union, y_union);
%                     union_area = polyarea(x_union(union_poly_ind), y_union(union_poly_ind));
%                     insect_area = polygon_intersect(bbox(1 : 2 : 8, ii)',bbox(2 : 2 : 8, ii)', ...
%                         bbox(1 : 2 : 8,jj)', bbox(2 : 2 : 8,jj)');
                    
%                     area_ii = polyarea(bbox(1 : 2 : 8, ii), bbox(2 : 2 : 8, ii));
%                     area_jj = polyarea(bbox(1 : 2 : 8, jj), bbox(2 : 2 : 8, jj));
%                     if(insect_area / min(area_ii, area_jj) > threshould)
%                         % score(ii)
%                         % score(jj)
%                         if (insect_area/min(area_ii, area_jj) > 0.9)
%                             if area_ii>area_jj
%                                 nms_flag(jj) = false;
%                             else
%                                 nms_flag(ii) = false;
%                                 break;
%                             end
%                         % elseif area_ii>area_jj
%                         %     nms_flag(jj) = false;
%                         % elseif score(ii)-score(jj)>score(jj)*0.1
%                         %     nms_flag(jj) = false;
%                         else
%                             nms_flag(jj) = false;
%                         end
%                     end
%                 end
%             end
%         end
%     end
% end

