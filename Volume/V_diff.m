function v = V_diff(c500, c250)
% Function of finding the volumetric difference between 2 polytopes 
    % by the equation: (A union B) - (A intersection B)

    vertices1 = readtable(c500);

    vertices2 = readtable(c250);

    vertices1 = table2array(vertices1); % 500 camel polytope
    vertices2 = table2array(vertices2); % 250 camel polytope
    [A1,b1]=vert2lcon(vertices1);
    [A2,b2]=vert2lcon(vertices2);
    vertices3 = lcon2vert([A1;A2], [b1;b2]); % intersection polytope

    vertices4 = [vertices1; vertices2]; % union polytope
    
    if isempty(vertices3)
        v = 0
    else
        % Intersection figure
        shp = alphaShape(vertices3,1);
        v1 = volume(shp);
        v1 = sprintf('%.6f',v1);

        % Camel 500 figure
        T = readtable(c500);
        x2 = T(:, {'x'});
        y2 = T(:, {'y'});
        z2 = T(:, {'z'});

        x2 = table2array(x2);
        y2 = table2array(y2);
        z2 = table2array(z2);
        P = [x2 y2 z2];
        shp2 = alphaShape(P,1);
        v2 = volume(shp2);
        v2 = sprintf('%.6f',v2);

        % Camel 250 figure
        T2 = readtable(c250);
        x3 = T2(:, {'x'});
        y3 = T2(:, {'y'});
        z3 = T2(:, {'z'});

        x3 = table2array(x3);
        y3 = table2array(y3);
        z3 = table2array(z3);

        P2 = [x3 y3 z3];
        shp3 = alphaShape(P2,1);
        v3 = volume(shp3);
        v3 = sprintf('%.6f',v3);

        % A Union B figure
        T3 = array2table(vertices4);
        x4 = T3(:, {'vertices41'});
        y4 = T3(:, {'vertices42'});
        z4 = T3(:, {'vertices43'});

        x4 = table2array(x4);
        y4 = table2array(y4);
        z4 = table2array(z4);

        P3 = [x4 y4 z4];
        shp4 = alphaShape(P3,1);
        v4 = volume(shp4);
        v4 = sprintf('%.6f',v4);


        % Calc volumetric difference
        v4 = convertCharsToStrings(v4);
        v4 = str2double(v4);
        v1 = convertCharsToStrings(v1);
        v1 = str2double(v1);
        v_diff = v4 - v1;
        v = v_diff;
    end
end
