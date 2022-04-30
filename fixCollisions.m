function modifiedTraj = fixCollisions(waypoints, output_table ,sv, numSamples)
trajectory = output_table(:,2:5);
alltimes = output_table(:,1);
length_of_traj = size(trajectory, 1);
%numSamples = 3000;

if length_of_traj > numSamples
    increment = round(length_of_traj/nSamples);
    trajectory = trajectory(1:increment:end,:);
end

xyz = trajectory(:, 2:4);
quat = angle2quat(trajectory(:, 4), zeros(numSamples,1), zeros(numSamples,1));

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
    %nSamples = 3000;
    minsnap_states = minsnappolytraj(waypoints',timepoints,nSamples,...
        MinSegmentTime=1e-4,MaxSegmentTime=10,TimeAllocation=true,TimeWeight=500)';    
    
    % Check if the new trajectory is valid
    valid = all(isStateValid(sv,minsnap_states));
   
end

xyz = minsnap_states(:, 1:3);
[yaw,pitch,roll] = quat2angle(minsnap_states(:,4:7));
validTraj = [xyz yaw];
modifiedTraj = [alltimes validTraj];

end