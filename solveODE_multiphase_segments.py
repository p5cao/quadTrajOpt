import openmdao.api as om
import dymos as dm
import numpy as np

from quadrotorODE import QuadrotorODE
from dymos.examples.plotting import plot_results
import matplotlib.pyplot as plt
from dymos.utils.testing_utils import assert_check_partials
from scipy.io import savemat, loadmat

# load the waypoints
# waypoints = loadmat('minsnaptraj_wps.mat')['minsnap_waypoints']
# waypoints = waypoints[0::2,:]
waypoints = loadmat('wps_from_path_planning.mat')['wps']
# row number of waypoints
num_of_wps = waypoints.shape[0]

# initial time
current_time = 0.0
current_cost = 0.0
time_array = np.array([0])
transcription = dm.Radau(num_segments=10, order = 3, compressed = False)
phase_dict ={}

i= 16

x_array = np.array([waypoints[i][0]])
y_array = np.array([waypoints[i][1]])
z_array = np.array([-waypoints[i][2]])
psi_array = np.array([waypoints[i][3]])

n1_array = np.array([250.0])
n2_array = np.array([250.0])
n3_array = np.array([250.0])
n4_array = np.array([250.0])

n1_0, n2_0, n3_0, n4_0 = n1_array[0], n2_array[0], n3_array[0], n4_array[0]
u0, v0, w0 = 0, 0, 0
phi0, theta0 = 0, 0


x0, x1 = waypoints[i][0], waypoints[i+1][0]
y0, y1 = waypoints[i][1], waypoints[i+1][1]
z0, z1 = -waypoints[i][2], -waypoints[i+1][2]
psi0, psi1 = waypoints[i][3], waypoints[i+1][3]

#psi0 = 0
p0, q0, r0 = 0, 0, 0
input_table = np.zeros(5)

# for i in range(num_of_wps-1):
for i in range(16, num_of_wps-1):
    
    
    # Instantiate the problem, add the driver, and allow it to use coloring
    p = om.Problem(model=om.Group())
    
    p.driver = om.pyOptSparseDriver()
    p.driver.declare_coloring()
    p.driver.options['optimizer'] = 'SNOPT'
    #p.driver.opt_settings['Major feasibility tolerance'] = 1e-9
    #p.driver.options['maxiter'] = 400 

    # Instantiate a Dymos Trajectory and add it to the Problem model.
    # Instantiate the trajectory and phase
    traj = dm.Trajectory()
    p.model.add_subsystem('traj', traj)
    phase_name = "phase{0}".format(i)
    traj_name = "traj_p{0}".format(i)
    phase_dict[i] = phase_name
    globals()[phase_name] = dm.Phase(ode_class=QuadrotorODE,
                 transcription=transcription)

    
    phase = traj.add_phase(phase_name, globals()[phase_name])
    # x0, x1 = waypoints[i][0], waypoints[i+1][0]
    # y0, y1 = waypoints[i][1], waypoints[i+1][1]
    # z0, z1 = -waypoints[i][2], -waypoints[i+1][2]
    # psi0, psi1 = waypoints[i][3], waypoints[i+1][3]
    
    # if abs(x1-x0)>4.0:
    #     ulimit = abs(x1-x0)/2.0
    # elif abs(x1-x0)<=1.0:
    #     ulimit = 1.0
    # else:
    ulimit = 3.0
        
    # if abs(y1-y0)>4.0:
    #     vlimit = abs(y1-y0)/2.0
    # elif abs(y1-y0)<=1.0:
    #     vlimit =1.0
    # else:
    vlimit = 3.0
    uref = ulimit*2/3
    vref = vlimit*2/3
    
    
    # add states
    if i == 0 :
        phase.set_time_options(fix_initial=True, units='s', duration_ref0 = 0, 
                                duration_bounds=(.5, 10.0),duration_ref =  6.0)
    else:
        phase.set_time_options(initial_bounds=(current_time-0.5, current_time+0.5), units='s', 
                               duration_ref0 = 0, duration_bounds=(.5, 10.0), duration_ref = 6.0)
    phase.add_state('x', fix_initial=True, fix_final=True, units='m', rate_source='xdot', 
                    lower = -5.0, upper = 50, ref0 = 0.0, ref = 40.0)
    
    phase.add_state('y', fix_initial=True, fix_final=True, units='m', rate_source='ydot',
                    lower = -18.0, upper = 15.0, ref0 = -15.0, ref = 15.0)
    
    phase.add_state('z', fix_initial=True, fix_final=True, units='m', rate_source='zdot',
                    lower = -2.2, upper = -1.8, ref0 = -2.2, ref = -1.8)

    phase.add_state('u', fix_initial=True, fix_final=False, units='m/s', rate_source='udot',
                    lower = -ulimit, upper = ulimit, ref0 = -uref, ref = uref)

    phase.add_state('v', fix_initial=True, fix_final=False, units='m/s', rate_source='vdot',
                    lower = -vlimit, upper = vlimit, ref0 = -vref, ref = vref)

    phase.add_state('w', fix_initial=True, fix_final=False, units='m/s', rate_source='wdot',
                    lower = -1.0, upper = 1.0, ref0 = -0.5, ref = 0.5)

    phase.add_state('phi', fix_initial=True, fix_final=False, units='rad', rate_source='phidot',
                    lower = -1/2*np.pi, upper = 1/2*np.pi, ref0 = -1/3*np.pi, ref = 1/3*np.pi)

    phase.add_state('theta', fix_initial=True, fix_final=False, units='rad', rate_source='thetadot',
                    lower = -1/2*np.pi, upper = 1/2*np.pi, ref0 = -1/3*np.pi, ref = 1/3*np.pi)

    phase.add_state('psi', fix_initial=True, fix_final=False, units='rad', rate_source='psidot',
                    lower = -np.pi+0.1, upper = np.pi-0.1, ref0 = -2/3*np.pi, ref = 2/3*np.pi)

    phase.add_state('p', fix_initial=True, fix_final=False, units='rad/s', rate_source='pdot',
                    lower = -1/2*np.pi/2, upper = 1/2*np.pi/2, ref0 = -1/3*np.pi/2, ref = 1/3*np.pi/2)
    
    phase.add_state('q', fix_initial=True, fix_final=False, units='rad/s', rate_source='qdot',
                    lower = -1/2*np.pi/2, upper = 1/2*np.pi/2, ref0 = -1/3*np.pi/2, ref = 1/3*np.pi/2)
    
    phase.add_state('r', fix_initial=True, fix_final=False, units='rad/s', rate_source='rdot',
                    lower = -1/2*np.pi, upper = 1/2*np.pi, ref0 = -1/3*np.pi, ref = 1/3*np.pi)

    # if i == 0 :
    #     phase.set_time_options(fix_initial=True, units='s', duration_ref0 = 0, 
    #                            duration_bounds=(.05, 2.0),duration_ref =  1.0)
    # else:
    #     phase.set_time_options(initial_bounds=(current_time-0.5, current_time+0.5), units='s', duration_ref0 = 0, 
    #                        duration_bounds=(.05, 2.0),duration_ref = 1.0)
    # phase.add_state('x', fix_initial=True, fix_final=True, units='m', rate_source='xdot', 
    #                 lower = -5.0, upper = 50, ref0 = 0.0, ref = 40.0)
    
    # phase.add_state('y', fix_initial=True, fix_final=True, units='m', rate_source='ydot',
    #                 lower = -18.0, upper = 15.0, ref0 = -15.0, ref = 15.0)
    
    # phase.add_state('z', fix_initial=True, fix_final=True, units='m', rate_source='zdot',
    #                 lower = -2.5, upper = -1.5, ref0 = -2.2, ref = -1.8)

    # phase.add_state('u', fix_initial=True, fix_final=False, units='m/s', rate_source='udot',
    #                 lower = -ulimit, upper = ulimit, ref0 = -uref, ref = uref)

    # phase.add_state('v', fix_initial=True, fix_final=False, units='m/s', rate_source='vdot',
    #                 lower = -vlimit, upper = vlimit, ref0 = -vref, ref = vref)

    # phase.add_state('w', fix_initial=True, fix_final=False, units='m/s', rate_source='wdot',
    #                 lower = -1.0, upper = 1.0, ref0 = -0.5, ref = 0.5)

    # phase.add_state('phi', fix_initial=True, fix_final=False, units='rad', rate_source='phidot',
    #                 lower = -1/2*np.pi, upper = 1/2*np.pi, ref0 = -1/3*np.pi, ref = 1/3*np.pi)

    # phase.add_state('theta', fix_initial=True, fix_final=False, units='rad', rate_source='thetadot',
    #                 lower = -1/2*np.pi, upper = 1/2*np.pi, ref0 = -1/3*np.pi, ref = 1/3*np.pi)

    # phase.add_state('psi', fix_initial=True, fix_final=True, units='rad', rate_source='psidot',
    #                 lower = -np.pi+0.1, upper = np.pi-0.1, ref0 = -2/3*np.pi, ref = 2/3*np.pi)

    # phase.add_state('p', fix_initial=True, fix_final=False, units='rad/s', rate_source='pdot',
    #                 lower = -1/3*np.pi/2, upper = 1/3*np.pi/2, ref0 = -1/3*np.pi/2, ref = 1/3*np.pi/2)
    
    # phase.add_state('q', fix_initial=True, fix_final=False, units='rad/s', rate_source='qdot',
    #                 lower = -1/2*np.pi/2, upper = 1/2*np.pi/2, ref0 = -1/3*np.pi/2, ref = 1/3*np.pi/2)
    
    # phase.add_state('r', fix_initial=True, fix_final=False, units='rad/s', rate_source='rdot',
    #                 lower = -1/2*np.pi, upper = 1/2*np.pi, ref0 = -1/3*np.pi, ref = 1/3*np.pi)

    
    phase.add_state('J', fix_initial=True, fix_final=False)

    # # inflow states
    # phase.add_state('nu0_1', fix_initial=False, fix_final=False, units=None, rate_source = 'nu0_1_dot')
    # phase.add_state('nuc_1', fix_initial=False, fix_final=False, units=None, rate_source = 'nuc_1_dot')
    # phase.add_state('nus_1', fix_initial=False, fix_final=False, units=None, rate_source = 'nus_1_dot')
    
    # phase.add_state('nu0_2', fix_initial=False, fix_final=False, units=None, rate_source = 'nu0_2_dot')
    # phase.add_state('nuc_2', fix_initial=False, fix_final=False, units=None, rate_source = 'nuc_2_dot')
    # phase.add_state('nus_2', fix_initial=False, fix_final=False, units=None, rate_source = 'nus_2_dot')
    
    # phase.add_state('nu0_3', fix_initial=False, fix_final=False, units=None, rate_source = 'nu0_3_dot')
    # phase.add_state('nuc_3', fix_initial=False, fix_final=False, units=None, rate_source = 'nuc_3_dot')
    # phase.add_state('nus_3', fix_initial=False, fix_final=False, units=None, rate_source = 'nus_3_dot')
    
    # phase.add_state('nu0_4', fix_initial=False, fix_final=False, units=None, rate_source = 'nu0_4_dot')
    # phase.add_state('nuc_4', fix_initial=False, fix_final=False, units=None, rate_source = 'nuc_4_dot')
    # phase.add_state('nus_4', fix_initial=False, fix_final=False, units=None, rate_source = 'nus_4_dot')
    
    # add controls
    phase.add_control('n1', units = '1/s', opt=True, lower = 150.0, upper = 450.0,
                       rate_continuity=True)
    phase.add_control('n2', units = '1/s', opt=True, lower = 150.0, upper = 450.0,
                       rate_continuity=True)
    phase.add_control('n3', units = '1/s', opt=True, lower = 150.0, upper = 450.0,
                       rate_continuity=True)
    phase.add_control('n4', units = '1/s', opt=True, lower = 150.0, upper = 450.0,
                       rate_continuity=True)

    # phase.add_control('F_z', units = 'N', opt=True, lower = 0, upper = 10,
    #                   rate_continuity=True)
    # phase.add_control('tau_x', units = 'N*m', opt=True, lower =-1, upper = 1,
    #                   rate_continuity=True)
    # phase.add_control('tau_y', units = 'N*m', opt=True, lower =-6, upper = 2,
    #                   rate_continuity=True)
    # phase.add_control('tau_z', units = 'N*m', opt=True, lower =-1, upper = 1, 
    #                   rate_continuity=True)
    # add outputs
    
    phase.add_timeseries_output('F_z', shape=(1,))
    phase.add_timeseries_output('tau_x', shape=(1,))
    phase.add_timeseries_output('tau_y', shape=(1,))
    phase.add_timeseries_output('tau_z', shape=(1,))
    '''minimum energy/ mechanical power'''
    phase.add_objective('J', loc = 'final', ref = 1.0)
    '''minimum time'''
    # phase.add_objective('time',  loc='final', ref=1.0)

    
    
    # x0, x1 = waypoints[i][0], waypoints[i+1][0]
    # y0, y1 = waypoints[i][1], waypoints[i+1][1]
    # z0, z1 = -waypoints[i][2], -waypoints[i+1][2]
    # psi1 = waypoints[i+1][3]
    
    
    
    # phase.add_boundary_constraint('x', loc='final', equals=x1)
    # phase.add_boundary_constraint('y', loc='final', equals=y1)
    # phase.add_boundary_constraint('z', loc='final', equals=z1)
    # phase.add_boundary_constraint('psi', loc='final', equals=psi1)
    # phase.add_boundary_constraint('F_z', loc = 'initial', equals = 6.762)
    # phase.add_boundary_constraint('tau_x', loc = 'initial', equals = 0.0)
    # phase.add_boundary_constraint('tau_y', loc = 'initial', equals = 0.0)
    # phase.add_boundary_constraint('tau_z', loc = 'initial', equals = 0.0)
    
    # if phase0 and phase 30, set all initial and final ns to 250, resp'ly
    if i == 0:
        
        phase.add_boundary_constraint('n1', loc = 'initial', equals = 250.0)
        phase.add_boundary_constraint('n2', loc = 'initial', equals = 250.0)
        phase.add_boundary_constraint('n3', loc = 'initial', equals = 250.0)
        phase.add_boundary_constraint('n4', loc = 'initial', equals = 250.0)
        
    # phase.add_boundary_constraint('x', loc='initial', equals=x0)
    # phase.add_boundary_constraint('y', loc='initial', equals=y0)
    # phase.add_boundary_constraint('z', loc='initial', equals=z0)
    # phase.add_boundary_constraint('psi', loc='initial', equals=psi0)
    # phase.add_boundary_constraint('u', loc='initial', equals=u0)
    # phase.add_boundary_constraint('v', loc='initial', equals=v0)
    # phase.add_boundary_constraint('w', loc='initial', equals=w0)
    # phase.add_boundary_constraint('phi', loc='initial', equals=phi0)
    # phase.add_boundary_constraint('theta', loc='initial', equals=theta0)
        # phase.add_objective('J', loc = 'final', ref = 1.0)
    if i == num_of_wps -1 :
    # if i == 2:
        phase.add_boundary_constraint('n1', loc = 'final', equals = 250.0)
        phase.add_boundary_constraint('n2', loc = 'final', equals = 250.0)
        phase.add_boundary_constraint('n3', loc = 'final', equals = 250.0)
        phase.add_boundary_constraint('n4', loc = 'final', equals = 250.0)
        
    
    # add waypoints BCs
    
    # if i > 0:
    #     last_phase_name = phase_dict[i-1]
    #     traj.link_phases(phases = [last_phase_name, phase_name], vars = ['*'])
    p.setup(check=True)

    p.set_val('traj.'+phase_name+'.t_initial', current_time, units='s')
    p.set_val('traj.'+phase_name+'.t_duration', current_time+10.0, units='s')

# p.model.linear_solver = om.DirectSolver()



#p.set_val('traj.phases.phase1.t_duration', 10.0, units='s')
# # for i in range(num_of_wps-1):
    
    
    # phase_name = phase_dict[i]
    if i != 16:
        x0 = x_array[-1][0]
        y0 = y_array[-1][0]
        z0 = z_array[-1][0]
        psi0 = psi_array[-1][0] 
        psi0 = (psi0+ np.pi) % (2 * np.pi) - np.pi
        x1 = waypoints[i+1][0]
        y1 = waypoints[i+1][1]
        z1 = -waypoints[i+1][2]
        psi1 = waypoints[i+1][3]

        


    p.set_val('traj.'+phase_name+'.states:x', phase.interp('x', [x0, x1]), units = 'm')
    p.set_val('traj.'+phase_name+'.states:y', phase.interp('y',[y0, y1]), units = 'm')
    p.set_val('traj.'+phase_name+'.states:z', phase.interp('z', [z0, z1]), units = 'm')

    p.set_val('traj.'+phase_name+'.states:u', phase.interp('u', [u0, 0]), units = 'm/s')
    p.set_val('traj.'+phase_name+'.states:v', phase.interp('v', [v0, 0]), units = 'm/s')
    p.set_val('traj.'+phase_name+'.states:w', phase.interp('w', [w0, 0]), units = 'm/s')
    p.set_val('traj.'+phase_name+'.states:phi', phase.interp('phi', [phi0, 0.0]), units = 'rad')
    p.set_val('traj.'+phase_name+'.states:theta', phase.interp('theta', [theta0, 0.0]), units = 'rad')
    p.set_val('traj.'+phase_name+'.states:psi', phase.interp('psi', [psi0, 0]), units = 'rad')
    p.set_val('traj.'+phase_name+'.states:p', phase.interp('p', [p0, 0.0]), units = 'rad/s')
    p.set_val('traj.'+phase_name+'.states:q', phase.interp('q', [q0, 0.0]), units = 'rad/s')
    p.set_val('traj.'+phase_name+'.states:r', phase.interp('r', [r0, 0.0]), units = 'rad/s')
    p.set_val('traj.'+phase_name+'.states:J', phase.interp('J', ys=[0, 300]), units = '1/s**2')
    # p.set_val('traj.phase0.states:nu0_1', phase.interp('nu0_1',[0.0, 0.0]), units=None)
    # p.set_val('traj.phase0.states:nuc_1', phase.interp('nuc_1',[0.0, 0.0]), units=None)
    # p.set_val('traj.phase0.states:nus_1', phase.interp('nus_1',[0.0, 0.0]), units=None)
    
    # p.set_val('traj.phase0.states:nu0_2', phase.interp('nu0_2',[0.0, 0.0]), units=None)
    # p.set_val('traj.phase0.states:nuc_2', phase.interp('nuc_2',[0.0, 0.0]), units=None)
    # p.set_val('traj.phase0.states:nus_2', phase.interp('nus_2',[0.0, 0.0]), units=None)
    
    # p.set_val('traj.phase0.states:nu0_3', phase.interp('nu0_3',[0.0, 0.0]), units=None)
    # p.set_val('traj.phase0.states:nuc_3', phase.interp('nuc_3',[0.0, 0.0]), units=None)
    # p.set_val('traj.phase0.states:nus_3', phase.interp('nus_3',[0.0, 0.0]), units=None)
    
    # p.set_val('traj.phase0.states:nu0_4', phase.interp('nu0_4',[0.0, 0.0]), units=None)
    # p.set_val('traj.phase0.states:nuc_4', phase.interp('nuc_4',[0.0, 0.0]), units=None)
    # p.set_val('traj.phase0.states:nus_4', phase.interp('nus_4',[0.0, 0.0]), units=None)
    # if i == 0:
    p.set_val('traj.'+phase_name+'.controls:n1',phase.interp('n1', ys=[250.0, 250.0]), units='1/s') 
    p.set_val('traj.'+phase_name+'.controls:n2',phase.interp('n2', ys=[250.0, 250.0]), units='1/s') 
    p.set_val('traj.'+phase_name+'.controls:n3',phase.interp('n3', ys=[250.0, 250.0]), units='1/s') 
    p.set_val('traj.'+phase_name+'.controls:n4',phase.interp('n4', ys=[250.0, 250.0]), units='1/s') 
    # else:
    #     p.set_val('traj.'+phase_name+'.controls:n1',phase.interp('n1', ys=[n1_0, 250.0]), units='1/s') 
    #     p.set_val('traj.'+phase_name+'.controls:n2',phase.interp('n2', ys=[n2_0, 250.0]), units='1/s') 
    #     p.set_val('traj.'+phase_name+'.controls:n3',phase.interp('n3', ys=[n3_0, 250.0]), units='1/s') 
    #     p.set_val('traj.'+phase_name+'.controls:n4',phase.interp('n4', ys=[n4_0, 250.0]), units='1/s') 
    
    p.run_model()
    # cpd = p.check_partials(method='cs', compact_print=True)
    # cpd = p.check_partials(compact_print=False, out_stream=None)
    # assert_check_partials(cpd, atol=1.0E-5, rtol=1.0E-4)
    #cpd = p.check_partials(method='cs', compact_print=True)
    
    # simulate the system without controls
    
    #Run the driver
    dm.run_problem(p, simulate=True)
    # p.run_driver()
    
    #sim_out = traj.simulate(times_per_seg=50)
    t_sol = p.get_val('traj.'+phase_name+'.timeseries.time')
    J = p.get_val('traj.'+phase_name+'.timeseries.states:J')
    
    


    sol = om.CaseReader('dymos_solution.db').get_case('final')
    sim = om.CaseReader('dymos_simulation.db').get_case('final')
    
    # F_z = sol.get_val('traj.phase0.timeseries.controls:F_z')
    # tau_x = sol.get_val('traj.phase0.timeseries.controls:tau_x')
    # tau_y = sol.get_val('traj.phase0.timeseries.controls:tau_y')
    # tau_z = sol.get_val('traj.phase0.timeseries.controls:tau_z')
    # u = sol.get_val('traj.phase0.timeseries.states:u')
    # v = sol.get_val('traj.phase0.timeseries.states:v')
    # w = sol.get_val('traj.phase0.timeseries.states:w')

    # # from scipy.io import savemat
    # # mdic_in = {"input_table": input_table, "label": "input_table"}
    # # savemat("input_table.mat", mdic_in)
    
    t_sim = sim.get_val('traj.'+phase_name+'.timeseries.time')
    x = sim.get_val('traj.'+phase_name+'.timeseries.states:x')
    y = sim.get_val('traj.'+phase_name+'.timeseries.states:y')
    z = sim.get_val('traj.'+phase_name+'.timeseries.states:z')
    u = sim.get_val('traj.'+phase_name+'.timeseries.states:u')
    v = sim.get_val('traj.'+phase_name+'.timeseries.states:v')
    w = sim.get_val('traj.'+phase_name+'.timeseries.states:w')
    psi = sim.get_val('traj.'+phase_name+'.timeseries.states:psi')
    theta = sim.get_val('traj.'+phase_name+'.timeseries.states:theta')
    phi = sim.get_val('traj.'+phase_name+'.timeseries.states:phi')
    p = sim.get_val('traj.'+phase_name+'.timeseries.states:p')
    q = sim.get_val('traj.'+phase_name+'.timeseries.states:q')
    r = sim.get_val('traj.'+phase_name+'.timeseries.states:r')
    
    current_time = t_sim[-1][0]
    current_cost = current_cost + J[-1][0]
    
    F_z = sim.get_val('traj.'+phase_name+'.timeseries.F_z')
    tau_x = sim.get_val('traj.'+phase_name+'.timeseries.tau_x')
    tau_y = sim.get_val('traj.'+phase_name+'.timeseries.tau_y')
    tau_z = sim.get_val('traj.'+phase_name+'.timeseries.tau_z')
    inputs = np.hstack((np.hstack((np.hstack((np.hstack((t_sim, F_z)), tau_x)), tau_y)), tau_z))
    input_table = np.vstack((input_table, inputs))
    
    x_array = np.vstack((x_array, x))
    y_array = np.vstack((y_array, y))
    z_array = np.vstack((z_array, z))
    psi_array = np.vstack((psi_array, psi))
    time_array = np.vstack((time_array, t_sim))
    
    n1 = sim.get_val('traj.'+phase_name+'.timeseries.controls:n1')
    n2 = sim.get_val('traj.'+phase_name+'.timeseries.controls:n2')
    n3 = sim.get_val('traj.'+phase_name+'.timeseries.controls:n3')
    n4 = sim.get_val('traj.'+phase_name+'.timeseries.controls:n4')
    
    n1_0 = n1[-1][0]
    n2_0 = n2[-1][0]
    n3_0 = n3[-1][0]
    n4_0 = n4[-1][0]  
    
    x0 = x[-1][0]
    y0 = y[-1][0]
    z0 = z[-1][0]
    u0 = u[-1][0]
    v0 = v[-1][0]
    w0 = w[-1][0]
    theta0 = theta[-1][0]
    phi0 = phi[-1][0]
    psi0 = psi[-1][0] 
    p0 = p[-1][0]
    q0 = q[-1][0]
    r0 = r[-1][0]
    
    # theta0 = (theta0+ np.pi) % (2 * np.pi) - np.pi
    # phi0 = (phi0+ np.pi) % (2 * np.pi) - np.pi
    # psi0 = (psi0+ np.pi) % (2 * np.pi) - np.pi
    
    n1_array = np.vstack((n1_array, n1))
    n2_array = np.vstack((n2_array, n2))
    n3_array = np.vstack((n3_array, n3))
    n4_array = np.vstack((n4_array, n4))
    
    error = np.linalg.norm([x0 - x1, y0 - y1])
    print("norm of difference = "+str(error) )
    if error > 10.0:
        print("optimization failed at "+phase_name)
        current_cost = current_cost - J[-1][0]
        current_time = current_time - (t_sim[-1][0] - t_sim[0][0])
        break

output_table = np.hstack((np.hstack((np.hstack((np.hstack((time_array, x_array)), y_array)), z_array)), psi_array))
output_table = np.delete(output_table, 0, 0)
# output_3to7 = np.hstack((np.hstack((np.hstack((np.hstack((time_array, x_array)), y_array)), z_array)), psi_array))

# mdic_out = {"output_all": output_table, "label": "output_table"}
# savemat("output_table_16to29.mat", mdic_out)
xline, yline, zline = output_table[:,1], output_table[:,2],-output_table[:,3]
# xline, yline, zline = x_array.flatten(), y_array.flatten(), -z_array.flatten()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(xline, yline, zline, 'gray')
# plot_results([('traj.phase1.timeseries.time', 'traj.phase1.timeseries.states:x',
#                 'time (s)', 'x (m)'),
#               ('traj.phase1.timeseries.time', 'traj.phase1.timeseries.states:y',
#                 'time (s)', 'y (m)'),
#               ('traj.phase1.timeseries.time', 'traj.phase1.timeseries.states:z',
#                 'time (s)', 'z (m)'),
#               ('traj.phase1.timeseries.time', 'traj.phase1.timeseries.states:psi',
#                 'time (s)', 'psi (rad)'),
#               ], 
             
#               title='Quadrotor outputs', 
             
#               p_sol=sol, p_sim = sim)

# plt.show()

# plot_results([('traj.phase1.timeseries.time', 'traj.phase1.timeseries.controls:n1',
#                                 'time (s)', 'n1 (revs/s)'),
#               ('traj.phase1.timeseries.time', 'traj.phase1.timeseries.controls:n2',
#                               'time (s)', 'n2 (revs/s)'),
#               ('traj.phase1.timeseries.time', 'traj.phase1.timeseries.controls:n3',
#                               'time (s)', 'n3 (revs/s)'),              
#               ('traj.phase1.timeseries.time', 'traj.phase1.timeseries.controls:n4',
#                                 'time (s)', 'n4 (revs/s)')],
#               title='Quadrotor control inputs', 
#               p_sol=sol, p_sim = sim)

# plt.show()

# # plot_results([('traj.phase0.timeseries.time', 'traj.phase0.timeseries.states:u',
# #   'time (s)', 'V_x (m/s)'),
# #               ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.states:v',
# #                               'time (s)', 'V_y (m/s)'),
# #               ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.states:w',
# #                               'time (s)', 'V_z (m/s)')],              
# #               title='Quadrotor velocity states', 
# #               p_sol=sol, p_sim = sim)

# # plt.show()


# plot_results([('traj.phase3.timeseries.time', 'traj.phase3.timeseries.states:psi',
#   'time (s)', 'yaw (rad)'),
#               ('traj.phase3.timeseries.time', 'traj.phase3.timeseries.states:theta',
#                               'time (s)', 'pitch (rad)'),
#               ('traj.phase3.timeseries.time', 'traj.phase3.timeseries.states:phi',
#                               'time (s)', 'roll (rad)')],              
#               title='Quadrotor attitude states', 
#               p_sol=sol, p_sim = sim)

# plt.show()


