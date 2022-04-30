clc
close all


rng(100,"twister")

% mapData = load("lab_occupancy_map.mat");
% omap = mapData.map3D;
ptCloud = pcread("lab3dMap.pcd");

omap = occupancyMap3D(5);
pose = [ 0 0 0 1 0 0 0];
maxRange = 50;
insertPointCloud(omap,pose,ptCloud,maxRange)

omap.FreeThreshold = omap.OccupiedThreshold;

%% create a binary occupancy map
resolution = 0.1;
XnumGrid = round((ptCloud.XLimits(2)-ptCloud.XLimits(1))/resolution)+1;
YnumGrid = round((ptCloud.YLimits(2)-ptCloud.YLimits(1))/resolution)+1;

XGrid = round(ptCloud.XLimits(1), 1):resolution:round(ptCloud.XLimits(2), 1);
YGrid = round(ptCloud.YLimits(1), 1):resolution:round(ptCloud.YLimits(2), 1);
Zcoords = 1.8:0.1:2.2;

map2d_h2m = zeros(YnumGrid,XnumGrid);

% assign binary occupancy

for i = 1:length(XGrid)
    x = XGrid(i);
    for j = 1:length(YGrid)
        y = YGrid(j);
        if map2d_h2m(j,i) == 1
            continue
        else
            for k = 1:length(Zcoords)
                z = Zcoords(k);
                if getOccupancy(omap,[x y z]) >= omap.OccupiedThreshold
                    map2d_h2m(j,i) = 1;
                end
            end
        end
    end
end


[X,Y] = meshgrid(XGrid,YGrid);
ct_map_2m = [X(map2d_h2m==1) Y(map2d_h2m==1)];
% UAV field of view is 1m x 1m
X_range = round(ptCloud.XLimits(1)):1.0:round(ptCloud.XLimits(2));
Y_range = round(ptCloud.YLimits(1)):1.0:round(ptCloud.YLimits(2));

grid_occupancy = zeros(length(X_range), length(Y_range));

for i = 1:length(X_range)
    grid_occupancy(1,i) = 1;
    for j = 2:length(Y_range)
        % corners and map boundaries are considered occupied
        if (i == 1 || j == 2 || j == length(Y_range) || i == length(X_range))
            occupancy = 100;
        else
            occupancy = sum(sum(bin_map((j-2)*10+7:(j-1)*10+7, (i-1)*10:i*10)));
        end
        if occupancy >=30
            grid_occupancy(j,i) = 1;
        end
    end
end
% figure
% imshow(~grid_occupancy)

[node_X,node_Y] = meshgrid(X_range,Y_range);
node_X = node_X';
node_Y = node_Y';
start_node = [5.0 0.0];

node_list = [start_node];
occupancy_list = [0];

for i = 1:length(X_range)
    for j = 1:length(Y_range)
        node = [node_X(i,j) node_Y(i,j)];
        node_list = [node_list;node];
        occupancy = grid_occupancy(j,i);
        occupancy_list = [occupancy_list;occupancy];
    end
end

% build the adjacency matrix

edge_list = [];
edge_costs = [];

adjacency = ones(length(node_list), length(node_list))*inf;
for i = 1:length(node_list)
    for j = 1:length(node_list)
        if occupancy_list(i) == 1 || occupancy_list(j) == 1
            continue
        else 
            if i==j
                adjacency(i,j) = 0;
                %continue
            else 
                if norm(node_list(i,:) - node_list(j,:)) == 1
                        adjacency(i,j) = 1;
                        %continue
                end
                if abs(norm(node_list(i,:) - node_list(j,:)) - sqrt(2)) < 0.01
                     adjacency(i,j) = sqrt(2);
                     %continue
                end
            end
        end
        if adjacency(i,j) == 1 || adjacency(i,j) == sqrt(2)
            edge_list = [edge_list; [i j]];
            edge_costs = [edge_costs; adjacency(i,j)];
        end
    end
end


figure
for ii = 1:length(edge_costs)
    plot([node_list(edge_list(ii,1), 1) node_list(edge_list(ii,2),1)],...
        [node_list(edge_list(ii,1), 2) node_list(edge_list(ii,2),2)])
    hold on
end

path_segments(1) = {[5 0; 2 6]};
path_segments(2) = {[2 6; 10 8]};
path_segments(3) = {[10 8; 11 4]};
path_segments(4) = {[11 4; 14 5]};
path_segments(5) = {[14 5;15 13]};
path_segments(6) = {[15 13; 19 11]};
path_segments(7) = {[19 11; 16 4]};
path_segments(8) = {[16 4; 19 6]};
path_segments(9) = {[19 6; 18 2]};
path_segments(10) = {[18 2; 20 1]};
path_segments(11) = {[20 1; 21 4]};
path_segments(12) = {[21 4; 23 2]};
path_segments(13) = {[23 2; 19 -3]};
path_segments(14) = {[19 -3; 25 0]};
path_segments(15) = {[25 0; 25 -6]};
path_segments(16) = {[25 -6; 27 -1]};
path_segments(17) = {[27 -1; 30 -4]};
path_segments(18) = {[30 -4; 33 -3]};
path_segments(19) = {[33 -3; 36 -6]};
path_segments(20) = {[36 -6; 32 -7]};
path_segments(21) = {[32 -7; 27 -6]};
path_segments(22) = {[27 -6; 25 -12]};
path_segments(23) = {[25 -12; 20 -17]};
path_segments(24) = {[20 -17; 23 -7]};
path_segments(25) = {[23 -7; 16 -11]};
path_segments(26) = {[16 -11; 22 -5]};
path_segments(27) = {[22 -5; 20 -3]};
path_segments(28) = {[20 -3; 17 -9]};
path_segments(29) = {[17 -9; 14 -3]};
path_segments(30) = {[14 -3; 18 -2]};
path_segments(31) = {[18 -2; 13  2]};
path_segments(32) = {[13  2; 12 -3]};
path_segments(33) = {[12 -3; 10 0]};
path_segments(34) = {[10 0; 6 4]};
path_segments(35) = {[6 4; 5 0]};

coverage_path = [];
for i = 1:length(path_segments)
    path = plan_A_star(path_segments{1,i}(1,:), path_segments{1,i}(2,:), ...
        edge_list, edge_costs, node_list);
    coverage_path = [coverage_path; path];
end

figure
%contour(X,Y,map2d_h2m)
scatter(ct_map_2m(:,1),ct_map_2m(:,2),'*')

hold on
plot(coverage_path(:,1), coverage_path(:,2)+ 0.5, 'Color','k', 'LineWidth',2)

cp_waypoints = []; % extract waypoints from the coverage path
alt = 2; %flying height is 2m agl 
zero_psi_univec = [1;0];

for ii = 1:size(coverage_path, 1)
    waypoint = zeros(1,4);
    xy = coverage_path(ii,:);
    if ii == size(coverage_path, 1)
        xy_next = coverage_path(1,:);
    else
        xy_next = coverage_path(ii+1,:);
    end
    z = alt;
    head_vec = xy_next-xy;
    if head_vec(1) ~= 0
        psi = atan(head_vec(2)/head_vec(1));
    else
       if head_vec(2)> 0
           psi = pi/2;
       else
           psi = -pi/2;
       end
    end
   
    
    if ii == 1 
        cp_waypoints = [cp_waypoints; waypoint];
    else
        xy_prev = coverage_path(ii-1,:);
%         if isnan(psi)
%             psi = 0;
%         end
        %psi = wrapToPi(psi);
        waypoint = [xy z psi];
        nx_waypoint = [xy_next z psi];

        if (abs(psi-last_psi) > pi/4) ...
                && ~any(find_node_idx_waypoint(xy, cp_waypoints(:,1:2)))
            cp_waypoints = [cp_waypoints; waypoint];
            %cp_waypoints = [cp_waypoints; nx_waypoint];
        end
        if ((xy_next-xy)*(xy-xy_prev)' == 0) ...
                && ~any(find_node_idx_waypoint(xy, cp_waypoints(:,1:2)))
            cp_waypoints = [cp_waypoints; waypoint];
            cp_waypoints = [cp_waypoints; nx_waypoint];
        end
        if ii < size(coverage_path, 1) -2
            psi_ahead = acos((coverage_path(ii+2,:)-xy)*zero_psi_univec/...
                (norm(coverage_path(ii+2,:)-xy)*norm(zero_psi_univec)));
            ahead_waypoint = [coverage_path(ii,:) z psi_ahead];
            if psi-psi_ahead == 0 && ~find_node_idx_waypoint(coverage_path(ii+3,:),...
                    cp_waypoints(:,1:2))
                cp_waypoints = [cp_waypoints;ahead_waypoint];
            end
        end
    end
    last_psi = psi;
end

for i = 1: size(cp_waypoints,1)
    xy = cp_waypoints(i, 1:2);
    if any(find_node_idx_waypoint(xy, ...
            cp_waypoints(setdiff(1:size(cp_waypoints,1),i), 1:2)))
        cp_waypoints(i, :) = zeros(1,4);
    end
%     if i > 1
%         prev_psi = cp_waypoints(i-1, 4);
%         cur_psi = cp_waypoints(i, 4);
%         if abs(cur_psi - prev_psi) >= 3/4*pi
%             cp_waypoints(i, :) = zeros(1,4);
%         end
%     end
end

zero_rows = all(cp_waypoints== zeros(1,4), 2);
cp_waypoints(zero_rows,:) = [];

plot(cp_waypoints(:,1), cp_waypoints(:,2)+ 0.5, 'Color','g', 'LineWidth',2)


wps =[ 5 0.5 2 -pi/2;
    2 6.5 2 atan(1/4); 
    10 8.5 2 atan(-4/-1);
    11 4.5 2 atan(1/3); 
    14 5.5 2 atan((13.5-5.5)/1);
    15 13.5 2 atan((11.5-13.5)/4);
    19 11.5 2 atan(-2/-2);
    17 9.5 2 atan((4.5-9.5)/-1);
    16 4.5 2 atan((2.5-4.5)/(23-16));
    23 2.5 2 atan((-2.5-2.5)/(19-23));
    19 -1.5 2 atan((0.5+2.5)/(25-19));
    25 0.5 2 -pi/2;
    25 -5.5 2 atan((-2.5+5.5)/(27-25));
    27 -2.5 2 atan((-3.5+2.5)/(30-27));
    30 -3.5 2 atan((-2.5+3.5)/(33-30));
    33 -2.5 2 atan((-5.5+2.5)/(36-33));
    36 -5.5 2 atan((-6.5+5.5)/(27-36));
    27 -6.5 2 atan((-16.5+6.5)/(20-27));
    20 -15 2 atan((-7.5+16.5)/(23-20));
    23 -7.5 2 atan((-10.5+7.5)/(16-23));
    16 -9.5 2 atan((-4.5+10.5)/(22-16));
    22 -4.5 2 atan((-2.5+4.5)/(20-22));
    20 -2.5 2 atan((-5.5+2.5)/(15-20));
    15 -5.5 2 atan((-1.5+5.5)/(18-15));
    18 -1.5 2 atan((2.5+1.5)/(13-18));
    13 2.5 2 atan((-2.5-2.5)/(12-13));
    12 -2.5 2 atan((0.5+2.5)/(10-12));
    10 0.5 2 atan((-0.5-0.5)/(8-10));
    8 -0.5 2 atan((4.5+0.5)/(6-8));
    6 4.5 2 atan((0.5-4.5)/(5-6));
    5 0.5 2 -pi/2];

%% Plot map and trajectories
figure
pcshow(ptCloud)
set(gcf,'color','w')
set(gca,'color','w')
xlabel("X(m)")
ylabel("Y(m)")
zlabel("Z(m)")

zlim([-0.1, 2.8])
hold on
plot3(wps(:,1), wps(:,2), wps(:,3), "-r", 'Linewidth',2)

%% plot quadtrajopt results
plot3(output_all(:, 2), output_all(:, 3), ...
    -output_all(:, 4), "-g", 'Linewidth',2)

%% plot LQR trajectory
hold on 
% plot3(third_sim_data_90s.Data(:,1), -third_sim_data_90s.Data(:,2), ...
%     -third_sim_data_90s.Data(:,3),"-g", 'Linewidth',2)

% compute mechanical energy needed

% run quadrotor_sys.slx 
wrench1 = out.wrench1;
LQR_mechE = wrench2energy(wrench1); % unit: Joules
fifth_sim_data_84s = out.trajectory;
initial_pts = [minsnap_waypoints(1:11,:); fifth_sim_data_84s.Data(440, 1)...
    -fifth_sim_data_84s.Data(440, 2) -fifth_sim_data_84s.Data(440, 3)...
    fifth_sim_data_84s.Data(440, 4)];

plot3([initial_pts(:,1); fifth_sim_data_84s.Data(440:end, 1)], [initial_pts(:,2); ...
    -fifth_sim_data_84s.Data(440:end, 2)], [initial_pts(:,3);...
    -fifth_sim_data_84s.Data(440:end, 3)], "-g", 'Linewidth',2)

view([-31 63])
legend("","Planned Coverage Path", "LQR Waypoint Following")
%% Generate Minimum Snap UAV Trajectory

% waypoints = cp_waypoints;
% nWayPoints = size(cp_waypoints, 1);
waypoints = wps;
nWayPoints = size(wps, 1);
% Calculate the distance between waypoints
distance = zeros(1,nWayPoints);
for i = 2:nWayPoints
    distance(i) = norm(waypoints(i,1:3) - waypoints(i-1,1:3));
end

% Assume a UAV speed of 3 m/s and calculate time taken to reach each waypoint
UAVspeed = 2.2;
timepoints = cumsum(distance/UAVspeed);
numSamples = 200;

[q,qd,qdd,qddd,qdddd,pp,timepoints,tsamples] = minsnappolytraj(waypoints',timepoints,numSamples,...
    MinSegmentTime=0.1,MaxSegmentTime=10,TimeAllocation=true,TimeWeight=100);

minsnap_output = q';
% check collisions
statespace = stateSpaceSE3([-3.0 37.0;
                    -18.0 14.0;
                     0.5 3;
                    -inf inf;
                    -inf inf;
                    -inf inf;
                    -inf inf]);

sv = validatorOccupancyMap3D(statespace,Map=omap);
sv.ValidationDistance = 0.1;
xyz = minsnap_output(:, 1:3);
quat = angle2quat(minsnap_output(:, 4), zeros(numSamples,1), zeros(numSamples,1));
minsnap_states = [xyz quat];

xyz_wp = waypoints(:, 1:3);
quat_wp = angle2quat(waypoints(:, 4), zeros(nWayPoints,1), zeros(nWayPoints,1));
waypoints = [xyz_wp quat_wp];

valid = all(isStateValid(sv,minsnap_states));

while(~valid)
    % Check the validity of the states
    validity = isStateValid(sv,minsnap_states);

    % Map the states to the corresponding waypoint segments
    segmentIndices = exampleHelperMapStatesToPathSegments(waypoints,minsnap_states);

    % Get the segments for the invalid states
    % Use unique, because multiple states in the same segment might be invalid
    invalidSegments = unique(segmentIndices(~validity));

    % Add intermediate waypoints on the invalid segments
    for i = 1:size(invalidSegments)
        segment = invalidSegments(i);
        if segment ~= 0
            % Take the midpoint of the position to get the intermediate position
            midpoint(1:3) = (waypoints(segment,1:3) + waypoints(segment+1,1:3))/2;
            
            % Spherically interpolate the quaternions to get the intermediate quaternion
            midpoint(4:7) = slerp(quaternion(waypoints(segment,4:7)),quaternion(waypoints(segment+1,4:7)),.5).compact;
            waypoints = [waypoints(1:segment,:); midpoint; waypoints(segment+1:end,:)];
        end
    end

    nWayPoints = size(waypoints,1);
    distance = zeros(1,nWayPoints);
    for i = 2:nWayPoints
        distance(i) = norm(waypoints(i,1:3) - waypoints(i-1,1:3));
    end
    
    % Calculate the time taken to reach each waypoint
    timepoints = cumsum(distance/UAVspeed);
    nSamples = 200;
    [q,qd,qdd,qddd,qdddd,pp,timepoints,tsamples] = minsnappolytraj(waypoints',timepoints,nSamples,...
        MinSegmentTime=1e-4,MaxSegmentTime=10,TimeAllocation=true,TimeWeight=500);    
    minsnap_states = q';
    minsnap_accels = qdd';
    % Check if the new trajectory is valid
    valid = all(isStateValid(sv,minsnap_states));
   
end

[yaw,pitch,roll ] = quat2angle( minsnap_states(:, 4:7));
minsnap_waypoints = [minsnap_states(:, 1:3) yaw];
minsnap_waypoints(1,:) = [];
minsnap_waypoints = [minsnap_waypoints; 5	0.500000000000000	2	-1.57079632679490];
% compute mechanical energy needed
minsnap_timetable = [tsamples' minsnap_waypoints];

% run quadrotor_sys.slx 
wrench = out.wrench;
minsnap_mechE = wrench2energy(wrench); % unit: Joules

plot3(minsnap_states(:, 1), minsnap_states(:, 2), ...
    minsnap_states(:, 3), "-y", 'Linewidth',2)
