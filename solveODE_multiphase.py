import openmdao.api as om
import dymos as dm
import numpy as np

from quadrotorODE import QuadrotorODE
from dymos.examples.plotting import plot_results
import matplotlib.pyplot as plt
from dymos.utils.testing_utils import assert_check_partials
from scipy.io import savemat, loadmat

# load the waypoints
waypoints = loadmat('wps_from_path_planning.mat')['wps']

# row number of waypoints
num_of_wps = waypoints.shape[0]
# Instantiate the problem, add the driver, and allow it to use coloring
p = om.Problem(model=om.Group())
p.driver = om.ScipyOptimizeDriver()
#p.driver = om.pyOptSparseDriver()
p.driver.declare_coloring()
p.driver.options['optimizer'] = 'SLSQP'
p.driver.options['maxiter']= 400


# Instantiate a Dymos Trajectory and add it to the Problem model.
# Instantiate the trajectory and phase
traj = dm.Trajectory()
p.model.add_subsystem('traj', traj)


transcription = dm.Radau(num_segments=10, order = 3, compressed = True)
phase_dict ={}

# for i in range(num_of_wps-1):
for i in range(3):
    phase_name = "phase{0}".format(i+1)
    traj_name = "traj_p{0}".format(i+1)
    phase_dict[i] = phase_name
    globals()[phase_name] = dm.Phase(ode_class=QuadrotorODE,
                 transcription=transcription)


    globals()[traj_name] = traj.add_phase(phase_name, globals()[phase_name])


    # add states
    globals()[traj_name].set_time_options(fix_initial=False, units='s', duration_bounds=(.05, 10.0),duration_ref0 = 0.0, duration_ref=10.0)
    
    globals()[traj_name].add_state('x', fix_initial=False, fix_final=True, units='m', rate_source='xdot', 
                    lower = 0, upper = 36.0, ref0 = 0.0, ref = 20.0)
    
    globals()[traj_name].add_state('y', fix_initial=False, fix_final=True, units='m', rate_source='ydot',
                    lower = -18.0, upper = 14.0, ref0 = -10.0, ref = 10.0)
    
    globals()[traj_name].add_state('z', fix_initial=False, fix_final=True, units='m', rate_source='zdot',
                    lower = -4.0, upper = 4.0, ref0 = -3.0, ref = 3.0)
    
    globals()[traj_name].add_state('u', fix_initial=True, fix_final=False, units='m/s', rate_source='udot',
                    lower = -5.0, upper = 5.0, ref0 = -3.0, ref = 3.0)
    
    globals()[traj_name].add_state('v', fix_initial=True, fix_final=False, units='m/s', rate_source='vdot',
                    lower = -4.0, upper = 4.0, ref0 = -3.0, ref = 3.0)
    
    globals()[traj_name].add_state('w', fix_initial=True, fix_final=False, units='m/s', rate_source='wdot',
                    lower = -1.0, upper = 1.0, ref0 = -1.0, ref = 1.0)
    
    globals()[traj_name].add_state('phi', fix_initial=True, fix_final=False, units='rad', rate_source='phidot',
                    lower = -1/3*np.pi, upper = 1/3*np.pi, ref0 = -1/6*np.pi, ref = 1/6*np.pi)
    
    globals()[traj_name].add_state('theta', fix_initial=True, fix_final=False, units='rad', rate_source='thetadot',
                    lower = -1/2*np.pi, upper = 1/2*np.pi, ref0 = -1/6*np.pi, ref = 1/6*np.pi)
    
    globals()[traj_name].add_state('psi', fix_initial=False, fix_final=True, units='rad', rate_source='psidot',
                    lower = -np.pi, upper = np.pi, ref0 = -np.pi/2, ref = np.pi/2)
    
    globals()[traj_name].add_state('p', fix_initial=True, fix_final=False, units='rad/s', rate_source='pdot',
                    lower = -1/3*np.pi/2, upper = 1/3*np.pi/2, ref0 = -1/6*np.pi/2, ref = 1/6*np.pi/2)
    
    globals()[traj_name].add_state('q', fix_initial=True, fix_final=False, units='rad/s', rate_source='qdot',
                    lower = -1/2*np.pi/2, upper = 1/2*np.pi/2, ref0 = -1/3*np.pi/2, ref = 1/3*np.pi/2)
    
    globals()[traj_name].add_state('r', fix_initial=True, fix_final=False, units='rad/s', rate_source='rdot',
                    lower = -1/2*np.pi/2, upper = 1/2*np.pi/2, ref0 = -1/3*np.pi/2, ref = 1/3*np.pi/2)
    
    globals()[traj_name].add_state('J', fix_initial=True, fix_final=False)

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
    globals()[traj_name].add_control('n1', units = '1/s', opt=True, lower = 200.0, upper = 450.0,
                       rate_continuity=True)
    globals()[traj_name].add_control('n2', units = '1/s', opt=True, lower = 200.0, upper = 450.0,
                       rate_continuity=True)
    globals()[traj_name].add_control('n3', units = '1/s', opt=True, lower = 200.0, upper = 450.0,
                       rate_continuity=True)
    globals()[traj_name].add_control('n4', units = '1/s', opt=True, lower = 200.0, upper = 450.0,
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
    
    globals()[traj_name].add_timeseries_output('F_z', shape=(1,))
    globals()[traj_name].add_timeseries_output('tau_x', shape=(1,))
    globals()[traj_name].add_timeseries_output('tau_y', shape=(1,))
    globals()[traj_name].add_timeseries_output('tau_z', shape=(1,))
    
    '''minimum time'''
    # phase.add_objective('time',  loc='final', ref=1.0)
    
    
    
    x0, x1 = waypoints[i][0], waypoints[i+1][0]
    y0, y1 = waypoints[i][1], waypoints[i+1][1]
    z0, z1 = -waypoints[i][2], -waypoints[i+1][2]
    psi0, psi1 = waypoints[i][3], waypoints[i+1][3]
    
    
    # globals()[phase_name].add_boundary_constraint('x', loc='final', equals=x1)
    # globals()[phase_name].add_boundary_constraint('y', loc='final', equals=y1)
    # globals()[phase_name].add_boundary_constraint('z', loc='final', equals=z1)
    #globals()[phase_name].add_boundary_constraint('psi', loc='final', equals=psi1)
    # phase.add_boundary_constraint('F_z', loc = 'initial', equals = 6.762)
    # phase.add_boundary_constraint('tau_x', loc = 'initial', equals = 0.0)
    # phase.add_boundary_constraint('tau_y', loc = 'initial', equals = 0.0)
    # phase.add_boundary_constraint('tau_z', loc = 'initial', equals = 0.0)
    
    # if phase0 and phase 30, set all initial and final ns to 250, resp'ly
    if i == 0:
        globals()[traj_name].set_time_options(fix_initial=True, units='s', duration_bounds=(.05, 10.0),duration_ref0 = 0.0, duration_ref=10.0)
        globals()[traj_name].add_boundary_constraint('n1', loc = 'initial', equals = 250.0)
        globals()[traj_name].add_boundary_constraint('n2', loc = 'initial', equals = 250.0)
        globals()[traj_name].add_boundary_constraint('n3', loc = 'initial', equals = 250.0)
        globals()[traj_name].add_boundary_constraint('n4', loc = 'initial', equals = 250.0)
        # globals()[phase_name].add_boundary_constraint('x', loc='initial', equals=x0)
        # globals()[phase_name].add_boundary_constraint('y', loc='initial', equals=y0)
        # globals()[phase_name].add_boundary_constraint('z', loc='initial', equals=z0)
        #globals()[phase_name].add_boundary_constraint('psi', loc='initial', equals=psi0)
        # globals()[traj_name].add_objective('J', loc = 'final', ref = 1.0)
    # if i == 30:
    if i == 2:
        globals()[traj_name].add_boundary_constraint('n1', loc = 'final', equals = 250.0)
        globals()[traj_name].add_boundary_constraint('n2', loc = 'final', equals = 250.0)
        globals()[traj_name].add_boundary_constraint('n3', loc = 'final', equals = 250.0)
        globals()[traj_name].add_boundary_constraint('n4', loc = 'final', equals = 250.0)
        '''minimum energy/ mechanical power'''
        globals()[traj_name].add_objective('J', loc = 'final', ref = 1.0)
    
    # add waypoints BCs
    
    if i > 0:
        last_phase_name = phase_dict[i-1]
        traj.link_phases(phases = [last_phase_name, phase_name], vars = ['*'])

# p.model.linear_solver = om.DirectSolver()

p.setup(check=True)
p.set_val('traj.phases.phase1.t_initial', 0.0, units='s')
#p.set_val('traj.phases.phase1.t_duration', 10.0, units='s')
# # for i in range(num_of_wps-1):
for i in range(3):

    
    # p.set_val('traj.phase0.t_duration', 10.0, units='s')
    
    phase_name = phase_dict[i]
    x0, x1 = waypoints[i][0], waypoints[i+1][0]
    y0, y1 = waypoints[i][1], waypoints[i+1][1]
    z0, z1 = -waypoints[i][2], -waypoints[i+1][2]
    psi0, psi1 = waypoints[i][3], waypoints[i+1][3]


    p.set_val('traj.phases.'+phase_name+'.states:x', globals()[phase_name].interp('x', [x0, x1]), units = 'm')
    p.set_val('traj.'+phase_name+'.states:y', globals()[phase_name].interp('y',[y0, y1]), units = 'm')
    p.set_val('traj.'+phase_name+'.states:z', globals()[phase_name].interp('z', [z0, z1]), units = 'm')

    p.set_val('traj.'+phase_name+'.states:u', globals()[phase_name].interp('u', [0, 0]), units = 'm/s')
    p.set_val('traj.'+phase_name+'.states:v', globals()[phase_name].interp('v', [0, 0]), units = 'm/s')
    p.set_val('traj.'+phase_name+'.states:w', globals()[phase_name].interp('w', [0, 0]), units = 'm/s')
    p.set_val('traj.'+phase_name+'.states:phi', globals()[phase_name].interp('phi', [0.0, 0.0]), units = 'rad')
    p.set_val('traj.'+phase_name+'.states:theta', globals()[phase_name].interp('theta', [0.0, 0.0]), units = 'rad')
    p.set_val('traj.'+phase_name+'.states:psi', globals()[phase_name].interp('psi', [psi0, psi1]), units = 'rad')
    p.set_val('traj.'+phase_name+'.states:p', globals()[phase_name].interp('p', [0.0, 0.0]), units = 'rad/s')
    p.set_val('traj.'+phase_name+'.states:q', globals()[phase_name].interp('q', [0.0, 0.0]), units = 'rad/s')
    p.set_val('traj.'+phase_name+'.states:r', globals()[phase_name].interp('r', [0.0, 0.0]), units = 'rad/s')
    p.set_val('traj.'+phase_name+'.states:J', globals()[phase_name].interp('J', ys=[0, 1]), units = '1/s**2')
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
    
    p.set_val('traj.'+phase_name+'.controls:n1',globals()[phase_name].interp('n1', ys=[250.0, 250.0]), units='1/s') 
    p.set_val('traj.'+phase_name+'.controls:n2',globals()[phase_name].interp('n2', ys=[250.0, 250.0]), units='1/s') 
    p.set_val('traj.'+phase_name+'.controls:n3',globals()[phase_name].interp('n3', ys=[250.0, 250.0]), units='1/s') 
    p.set_val('traj.'+phase_name+'.controls:n4',globals()[phase_name].interp('n4', ys=[250.0, 250.0]), units='1/s') 


# p.set_val('traj.phase0.controls:F_z', phase.interp('F_z', ys=[6.762, 6.762]), units='N')
# p.set_val('traj.phase0.controls:tau_x', phase.interp('tau_x', ys=[0.0, 0.0]), units='N*m')
# p.set_val('traj.phase0.controls:tau_y', phase.interp('tau_y', ys=[0.0, 0.0]), units='N*m')
# p.set_val('traj.phase0.controls:tau_z', phase.interp('tau_z', ys=[0.0, 0.0]), units='N*m')

p.run_model()
cpd = p.check_partials(method='cs', compact_print=True)
# cpd = p.check_partials(compact_print=False, out_stream=None)
# assert_check_partials(cpd, atol=1.0E-5, rtol=1.0E-4)


# simulate the system without controls

#Run the driver
dm.run_problem(p, simulate=True)
# p.run_driver()

#sim_out = traj.simulate(times_per_seg=50)
t_sol = p.get_val('traj.phases.phase1.timeseries.time')
J = p.get_val('traj.phases.phase1.timeseries.states:J')
#t_sim = sim_out.get_val('traj.phase0.timeseries.time')

# states =['x', 'y', 'z', 'psi']

# fig, axes = plt.subplots(len(states), 1)
# for i, state in enumerate(states):
#     sol = axes[i].plot(t_sol, p.get_val(f'traj.phase0.timeseries.states:{state}'), 'o')
#     #sim = axes[i].plot(t_sim, sim_out.get_val(f'traj.phase0.timeseries.states:{state}'), '-')
#     axes[i].set_ylabel(state)
# axes[-1].set_xlabel('time (s)')
# #fig.legend((sol[0], sim[0]), ('solution', 'simulation'), 'lower right', ncol=2)
# plt.tight_layout()
# plt.show()


sol = om.CaseReader('dymos_solution.db').get_case('final')
sim = om.CaseReader('dymos_simulation.db').get_case('final')

# F_z = sol.get_val('traj.phase0.timeseries.controls:F_z')
# tau_x = sol.get_val('traj.phase0.timeseries.controls:tau_x')
# tau_y = sol.get_val('traj.phase0.timeseries.controls:tau_y')
# tau_z = sol.get_val('traj.phase0.timeseries.controls:tau_z')
# u = sol.get_val('traj.phase0.timeseries.states:u')
# v = sol.get_val('traj.phase0.timeseries.states:v')
# w = sol.get_val('traj.phase0.timeseries.states:w')

t_sim = sim.get_val('traj.phase1.timeseries.time')
F_z = sim.get_val('traj.phase1.timeseries.F_z')
tau_x = sim.get_val('traj.phase1.timeseries.tau_x')
tau_y = sim.get_val('traj.phase1.timeseries.tau_y')
tau_z = sim.get_val('traj.phase1.timeseries.tau_z')
input_table = np.hstack((np.hstack((np.hstack((np.hstack((t_sim, F_z)), tau_x)), tau_y)), tau_z))

# from scipy.io import savemat
# mdic_in = {"input_table": input_table, "label": "input_table"}
# savemat("input_table.mat", mdic_in)


x = sim.get_val('traj.phase1.timeseries.states:x')
y = sim.get_val('traj.phase1.timeseries.states:y')
z = sim.get_val('traj.phase1.timeseries.states:z')
psi = sim.get_val('traj.phase1.timeseries.states:psi')

output_table = np.hstack((np.hstack((np.hstack((np.hstack((t_sim, x)), y)), z)), psi))

# mdic_out = {"output_table": output_table, "label": "output_table"}
# savemat("output_table.mat", mdic_out)
                         
plot_results([('traj.phase1.timeseries.time', 'traj.phase1.timeseries.states:x',
                'time (s)', 'x (m)'),
              ('traj.phase1.timeseries.time', 'traj.phase1.timeseries.states:y',
                'time (s)', 'y (m)'),
              ('traj.phase1.timeseries.time', 'traj.phase1.timeseries.states:z',
                'time (s)', 'z (m)'),
              ('traj.phase1.timeseries.time', 'traj.phase1.timeseries.states:psi',
                'time (s)', 'psi (rad)'),
              ], 
             
              title='Quadrotor outputs', 
             
              p_sol=sol, p_sim = sim)

plt.show()

# plot_results([('traj.phase0.timeseries.time', 'traj.phase0.timeseries.controls:n1',
#                                'time (s)', 'n1 (revs/s)'),
#               ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.controls:n2',
#                               'time (s)', 'n2 (revs/s)'),
#               ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.controls:n3',
#                               'time (s)', 'n3 (revs/s)'),              
#               ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.controls:n4',
#                                'time (s)', 'n4 (revs/s)')],
#               title='Quadrotor control inputs', 
#               p_sol=sol, p_sim = sim)

# plt.show()

# plot_results([('traj.phase0.timeseries.time', 'traj.phase0.timeseries.states:u',
#   'time (s)', 'V_x (m/s)'),
#               ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.states:v',
#                               'time (s)', 'V_y (m/s)'),
#               ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.states:w',
#                               'time (s)', 'V_z (m/s)')],              
#               title='Quadrotor velocity states', 
#               p_sol=sol, p_sim = sim)

# plt.show()


plot_results([('traj.phase1.timeseries.time', 'traj.phase1.timeseries.states:psi',
  'time (s)', 'yaw (rad)'),
              ('traj.phase1.timeseries.time', 'traj.phase1.timeseries.states:theta',
                              'time (s)', 'pitch (rad)'),
              ('traj.phase1.timeseries.time', 'traj.phase1.timeseries.states:phi',
                              'time (s)', 'roll (rad)')],              
              title='Quadrotor attitude states', 
              p_sol=sol, p_sim = sim)

plt.show()


