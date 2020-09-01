classdef FigureRotator < handle

% FigureRotator
%
% Apply this to a figure to use a mouse to rotate around a target position
% and zoom in and out. This is similar to the Rotate 3D capability found in
% the standard figure menu, but this allows greater flexibility and
% fluidity of movement. An example is provided in example_figure_rotator.m,
% and additional examples are provided below.
%
% Left-click and drag to rotate about the target.
% Scroll the mouse wheel to move towards/away from the target.
% Right-click and drag to zoom the camera in/out, changing the view angle.
% Double-click to reset the "up" vector.
% Press 'r' to reset the view to what it was when the FigureRotator started.
%
% Example:
%
% figure();
% plot3(randn(1, 10), randn(1, 10), randn(1, 10));
% drawnow();
% f = FigureRotator(gca);
%
% The FigureRotator can later be stopped by calling the Stop() function.
%
% f.Stop();
%
% It's often helpful to specify the initial camera parameters, like
% position, target, up vector, and view angle. These can all be passed to
% the constructor. (These are all properties of axes objects in MATLAB. See 
% 'axes properties' in the documentation for more.)
%
% f = FigureRotator(gca, 'CameraTarget',    [0 0 0], ...
%                        'CameraPosition',  [15 0 0], ...
%                        'CameraUpVector',  [0 0 1], ...
%                        'CameraViewAngle', 60);
%
% The FigureRotator allows complete 3D rotation, so if you start losing
% track of "up", you can always re-align the camera's up vector with the 
% axes' [0 0 1] by calling RestoreUp().
% 
% You can also set the up vector with SetUpVector(). This take two arguments.
% The first is the 3D up vector, e.g., [0 0 1]. The second says whether "up"
% should always remain "up" (constrainted rotation).
%
% % Look at peaks sideways.
% peaks();
% f = FigureRotator();
% f.SetUpVector([0 1 0]);
% 
% Now try rotating and then double-clicking. Note how double-clicking realigns
% the y axis with up.
%
% Now try:
%
% f.SetUpVector([0 1 0], true); % Keep "up" pointing up.
%
% Note how it now rotates about the y axis.
%
% This object uses the figure's WindowButtonUpFcn, WindowButtonDownFcn,
% WindowButtonMotionFcn, WindowScrollWheelFcn, and KeyPressFcn callbacks. If 
% those are necessary for other tasks as well, callbacks can be attached to the
% FigureRotator, which will pass all arguments on to the provided callback
% function.
%
% Example:
%
% f = FigureRotator(gca);
% f.AttachCallback('WindowButtonDownFcn', 'disp(''clicked'');');
% 
% Or multiple callbacks can be set with a single call:
%
% f.AttachCallback('WindowButtonDownFcn',   'disp(''down'');', ...
%                  'WindowButtonUpFcn',     'disp(''up'');', ...
%                  'WindowButtonMotionFcn', 'disp(''moving'');', ...
%                  'WindowScrollWheelFcn',  @(~, ~) disp('scrolling'), ...
%                  'KeyPressFcn',           'disp(''key'');');
%
% A single FigureRotator can control multiple axes, even axes across
% multiple figures.
%
% Example:
% 
% figure(1);
% clf();
% ha1 = subplot(2, 1, 1);
% peaks;
% ha2 = subplot(2, 1, 2);
% peaks;
% 
% figure(2);
% clf();
% peaks;
% ha3 = gca();
% 
% f = FigureRotator([ha1 ha2 ha3]);
% 
% The FigureRotator now controls all three figures -- useful for keeping the
% same perspective across multiple objects.
%
% --- Change Log ---
% 
% 2014-09-11: Updated for R2014B graphics. Change copyright 2014 
% Tucker McClure.
%
% 2014-03-26: Allows view to be reset with 'r', has improved handling of
% callbacks for multiple axes, more examples of using up vector, and more
% comments. Change copyright 2014 Tucker McClure.
% 
% Original. Copyrirght 2012, The MathWorks, Inc.
% 
% ---
% 
% Copyright 2014, The MathWorks, Inc. and Tucker McClure
    
    properties
        
        % Figure and axes handles
        h_f;
        h_a;
        
        % Current states
        rotating = false;
        zooming  = false;

        % Mouse positions
        rotate_start_point = [0 0]; % Mouse position on button down
        zoom_start_point   = [0 0]; % Mouse position on button down
        
        wbdf; % Pass-through WindowButtonDownFcn
        wbuf; % Pass-through WindowButtonUpFcn
        wbmf; % Pass-through WindowButtonMotionFcn
        wswf; % Pass-through WindowScrollWheelFcn
        kpf;  % Pass-through KeyPressFcn
        
        keep_up = false; % True iff we should always restore up after movement
        up = [0; 0; 1];      % Up vector
        
        is_stopped = false; % True iff we should no longer control the axes.
        
        % Original states
        original;
        
    end
    
    methods
        
        % Construct a FigureRotator for the given axes.
        function o = FigureRotator(axes_handle, varargin)

            % Use gca if none is given.
            if nargin == 0
                axes_handle = gca();
            end
            
            % Record the axes and figure.
            o.h_a = axes_handle;
            o.h_f = get(axes_handle, 'Parent');
            if iscell(o.h_f)
                o.h_f = [o.h_f{:}];
            end

            % Pass any arguments on to the axes object.
            if nargin >= 2
                set(o.h_a, varargin{:});
            end
            
            % Get the original view positions.
            o.original.position   = get(o.h_a, 'CameraPosition');
            o.original.target     = get(o.h_a, 'CameraTarget');
            o.original.up         = get(o.h_a, 'CameraUpVector');
            o.original.view_angle = get(o.h_a, 'CameraViewAngle');
            
            % Save the original up vector as the current up vector.
            o.up = o.original.up;
            
            % Set the figure callbacks to register with this object.
            set(o.h_f, ...
                'WindowButtonDownFcn',   @o.ButtonDown, ...
                'WindowButtonUpFcn',     @o.ButtonUp, ...
                'WindowButtonMotionFcn', @o.Move, ...
                'WindowScrollWheelFcn',  @o.Wheel, ...
                'KeyPressFcn',           @o.Key);
            
            % Set up the axes object for what we need. We get the last
            % word.
            set(o.h_a, ...
                'CameraPositionMode',  'manual', ...
                'CameraTargetMode',    'manual', ...
                'CameraUpVectorMode',  'manual', ...
                'CameraViewAngleMode', 'manual', ...
                'XLimMode',            'manual', ...
                'YLimMode',            'manual', ...
                'ZLimMode',            'manual', ...
                'DataAspectRatioMode', 'manual');
            
        end
        
        % Called when a user clicks
        function ButtonDown(o, h, event, varargin)

            % If the user is clicking in the figure, but not on one of our axes,
            % ignore it.
            if ~any(ishandle(o.h_a)) || ~any(gca() == o.h_a)
                return;
            end
            
            % Get the button type.
            switch get(h, 'SelectionType')
                
                % Rotate around.
                case {'normal', 'extend'}

                    % Record the starting point and that we're rotating.
                    o.rotate_start_point = get(h, 'CurrentPoint');
                    o.rotating           = true;

                % Zoom.
                case 'alt'
                
                    % Record the starting point and that we're zooming.
                    o.zoom_start_point = get(h, 'CurrentPoint');
                    o.zooming          = true;
                    
                % When double-clicking, restore up.
                case 'open'
                    
                    o.RestoreUp();
                    
            end

            % If there's a callback attachment, execute it.
            execute_callback(o.wbdf, h, event, varargin{:});
            
        end
        
        % Called when user releases a click
        function ButtonUp(o, h, event, varargin)
            
            % If the user is clicking in the figure, but not on one of our axes,
            % ignore it.
            if ~any(ishandle(o.h_a)) || ~any(gca() == o.h_a)
                return;
            end
            
            % Get the button type.
            switch get(h, 'SelectionType')
                
                % Stop rotating.
                case {'normal', 'extend'}
                    o.rotating = false;
                    
                % Stop zooming.
                case 'alt'
                    o.zooming = false;
                    
            end
            
            % If there's a callback attachment, execute it.
            execute_callback(o.wbuf, h, event, varargin{:});
            
        end
        
        % Called when mouse moves in figure
        function Move(o, h, event, varargin)
            
            % If the user is clicking in the figure, but not on one of our axes,
            % ignore it.
            if ~any(ishandle(o.h_a)) || ~any(gca() == o.h_a)
                return;
            end
            
            if o.rotating
                
                % Get the mouse position in the window.
                s = feval(@(x) x(3:4), get(h, 'Position'));
                p = get(h, 'CurrentPoint');
                r = (p - o.rotate_start_point)./s;
                   
                % Get the current state wrt the target and frame.
                dar       = get(gca(), 'DataAspectRatio')';
                r_t0      = get(gca(), 'CameraTarget')'   ./ dar;
                r_c0      = get(gca(), 'CameraPosition')' ./ dar;
                up_hat    = get(gca(), 'CameraUpVector')' ./ dar;
                r_tc      = r_t0 - r_c0;
                r_tc_hat  = normalize(r_tc);
                
                % Correct up.
                up_hat    = normalize(up_hat - r_tc_hat'*up_hat*r_tc_hat);
                
                % Find "right" (this will be a unit vector since r_tc_hat
                % and up_hat are orthonormal).
                right_hat = cross(r_tc_hat, up_hat);
                
                % Calculate where the mouse is in the axes space from its
                % location in the 2D figure window.
                r_mc = r(2) * up_hat + r(1) * right_hat;
                
                % Calculate the rotation axis.
                a_hat = normalize(cross(r_tc - r_mc, r_tc));
                
                % Calculate the rotation matrix.
                Q = aa2dcm(a_hat, norm(r_mc)*pi);
                
                % Calculate the new camera position, accounting for
                % non-equal aspect ratios.
                r_n0 = -Q*r_tc + r_t0;
                
                % Update the relevant quantities.
                for k = 1:length(o.h_a)
                    if ishandle(o.h_a(k))
                        set(o.h_a(k), 'CameraPosition', r_n0 .* dar, ...
                                      'CameraUpVector', Q*up_hat .* dar);
                    end
                end
                          
                % If up should stay up at all time, restore it.
                if o.keep_up
                    o.RestoreUp();
                end
                       
                % Update the "last" accounted point.
                o.rotate_start_point = p;

            end
            
            if o.zooming
                
                % Get the starting view angle.
                view_angle = get(gca(), 'CameraViewAngle');
                
                % Get the mouse position in the window.
                s = feval(@(x) x(4), get(h, 'Position'));
                p = get(h, 'CurrentPoint');
                r = -(p(2) - o.zoom_start_point(2))/s;
                new_view_angle = min(2^r*view_angle, 180-eps);
                
                for k = 1:length(o.h_a)
                    if ishandle(o.h_a(k))
                        set(o.h_a(k), 'CameraViewAngle', new_view_angle);
                    end
                end
                
                % Update the "last" accounted point.
                o.zoom_start_point = p;
                
            end
            
            % If there's a callback attachment, execute it.
            execute_callback(o.wbmf, h, event, varargin{:});
            
        end
        
        % Called for scroll wheel
        function Wheel(o, h, event, varargin)
            
            % If the user is clicking in the figure, but not on one of our axes,
            % ignore it.
            if ~any(ishandle(o.h_a)) || ~any(gca() == o.h_a)
                return;
            end
            
            % Scalar to increase/decrease distance to target.
            s = 1.2^double(event.VerticalScrollCount);
            
            % Update what we're currently seeing by the appropriate amount.
            t0 = get(gca(), 'CameraTarget');
            c0 = get(gca(), 'CameraPosition'); 
            r_n0 = s * (c0 - t0) + t0;
            
            for k = 1:length(o.h_a)
                if ishandle(o.h_a(k))
                    set(o.h_a(k), 'CameraPosition', r_n0);
                end
            end
            
            % If there's a callback attachment, execute it.
            execute_callback(o.wswf, h, event, varargin{:});
            
        end
        
        % Called when a key is pressed in the figure.
        function Key(o, h, event, varargin)
            
            % If the user is clicking in the figure, but not on one of our axes,
            % ignore it.
            if ~any(ishandle(o.h_a)) || ~any(gca() == o.h_a)
                return;
            end
            
            % See which key.
            switch event.Key
                
                % If 'r', let's reset the view.
                case 'r'
                    if length(o.h_a) == 1
                        if ishandle(o.h_a)
                            set(o.h_a, ...
                                'CameraTarget',    o.original.target, ...
                                'CameraPosition',  o.original.position, ...
                                'CameraUpVector',  o.original.up, ...
                                'CameraViewAngle', o.original.view_angle);
                        end
                    else
                        for k = 1:length(o.h_a)
                            if ishandle(o.h_a(k))
                                set(o.h_a(k), ...
                                  'CameraTarget',    o.original.target{k}, ...
                                  'CameraPosition',  o.original.position{k}, ...
                                  'CameraUpVector',  o.original.up{k}, ...
                                  'CameraViewAngle', o.original.view_angle{k});
                            end
                        end
                    end
                           
            end
            
            % If there's a callback, execute it.
            execute_callback(o.kpf, h, event, varargin{:});
            
        end
        
        % Sometime users like to return "up" to [0 0 1], so we'll give them
        % a function to call.
        function RestoreUp(o)
            if iscell(o.up)
                for k = 1:length(o.up)
                    if ishandle(o.h_a(k))
                        set(o.h_a(k), 'CameraUpVector', o.up{k});
                    end
                end
            else
                for k = 1:length(o.h_a)
                    if ishandle(o.h_a(k))
                        set(o.h_a(k), 'CameraUpVector', o.up);
                    end
                end
            end
        end

        % Add a pass-through callback for one of the callbacks
        % FigureRotator hogs to itself. This way, a user can still get all
        % the info he needs from a figure's callbacks *and* use the 
        % rotator.
        function AttachCallback(o, varargin)
            
            for k = 2:2:length(varargin)
                switch varargin{k-1}
                    case 'WindowButtonDownFcn'
                        o.wbdf = varargin{k};
                    case 'WindowButtonUpFcn'
                        o.wbuf = varargin{k};
                    case 'WindowButtonMotionFcn'
                        o.wbmf = varargin{k};
                    case 'WindowScrollWheelFcn'
                        o.wswf = varargin{k};
                    case 'KeyPressFcn'
                        o.kpf  = varargin{k};
                    otherwise
                        warning('Invalid callback attachment.');
                end
            end
            
        end
        
        % Allow the user to specify that up should always be up and to
        % specify what up is.
        function SetUpVector(o, up, on)
            o.up = up;
            if nargin >= 3
                o.keep_up = logical(on);
            end
            o.RestoreUp();
        end
        
        % We're done. Get rid of the callbacks. If there were pass-through
        % callbacks, replace our callbacks with those.
        function Stop(o)
            o.is_stopped = true;
            for k = 1:length(o.h_f)
                if ishandle(o.h_f(k))
                    set(o.h_f(k), ...
                        'WindowButtonDownFcn',   o.wbdf, ...
                        'WindowButtonUpFcn',     o.wbuf, ...
                        'WindowButtonMotionFcn', o.wbmf, ...
                        'WindowScrollWheelFcn',  o.wswf, ...
                        'KeyPressFcn',           o.kpf);
                end
            end
        end
        
    end
    
end

% Safely normalize an input vector.
function x_hat = normalize(x)
    n = norm(x);
    if n > eps
        x_hat = x/n;
    else
        x_hat = x;
    end
end

% Convert the specified axis and angle of rotation to a direction cosine
% matrix.
function M = aa2dcm(ax, an)
    M = eye(3)*cos(an) + (1-cos(an))*(ax*ax') + crs(ax)*sin(an);
end

% Returns a skew-symmetric "cross product" matrix from 3-by-1 vector, v,
% such that cross(v, b) == crs(v)*b.
function M = crs(v)
    M = [ 0    -v(3)  v(2); ...
          v(3)  0    -v(1); ...
         -v(2)  v(1)  0];
end

% Execute whatever callback was requested.
function execute_callback(cb, h, event, varargin)

    % If there's anything here...
    if ~isempty(cb)
        
        % If might be a regular function handle. If so, just pass along the
        % handle and event.
        if isa(cb, 'function_handle')
            
            cb(h, event);
            
        % If it's a cell array, it should contain a function handle and 
        % additional arguments.
        elseif iscell(cb)
            
            cb(h, event, varargin{:});
            
        % Otherwise, if it's text, evaluate it.
        elseif ischar(cb) && ~isempty(cb)
            
            eval(cb);
            
        % Otherwise, we don't know what to do.
        else
            
            error('FigureRotator:InvalidCallback', ...
                  'Invalid figure callback in FigureRotator.');
            
        end
        
    end
    
end


