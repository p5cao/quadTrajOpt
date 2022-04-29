import openmdao.api as om
import dymos as dm
import numpy as np

from quadrotorODE import QuadrotorODE
from dymos.examples.plotting import plot_results
import matplotlib.pyplot as plt
from dymos.utils.testing_utils import assert_check_partials
from scipy.io import savemat

# Instantiate the problem, add the driver, and allow it to use coloring
p = om.Problem(model=om.Group())
#p.driver = om.ScipyOptimizeDriver()
p.driver = om.pyOptSparseDriver()
p.driver.declare_coloring()
p.driver.options['optimizer'] = 'SNOPT'
p.driver.opt_settings['Major feasibility tolerance'] = 1e-9

# Instantiate a Dymos Trajectory and add it to the Problem model.
# Instantiate the trajectory and phase
traj = dm.Trajectory()

phase = dm.Phase(ode_class=QuadrotorODE,
                 transcription=dm.Radau(num_segments=10, order = 3, compressed = True))


traj.add_phase('phase0', phase)

p.model.add_subsystem('traj', traj)

# add states
phase.set_time_options(fix_initial=True, units='s', duration_bounds=(.05, 10.0),duration_ref0 = 0.0, duration_ref=10.0)

phase.add_state('x', fix_initial=True, fix_final=True, units='m', rate_source='xdot', 
                lower = -5.0, upper = 40.0, ref0 = 0.0, ref = 20.0)

phase.add_state('y', fix_initial=True, fix_final=True, units='m', rate_source='ydot',
                lower = -20.0, upper = 15.0, ref0 = -2.0, ref = 1.0)

phase.add_state('z', fix_initial=True, fix_final=True, units='m', rate_source='zdot',
                lower = -4.0, upper = 0.0, ref0 = -2.0, ref = 1.0)

phase.add_state('u', fix_initial=True, fix_final=False, units='m/s', rate_source='udot',
                lower = -1.0, upper = 5.0, ref0 = 0.0, ref = 4.0)

phase.add_state('v', fix_initial=True, fix_final=False, units='m/s', rate_source='vdot',
                lower = -3.0, upper = 3.0, ref0 = -1.0, ref = 1.0)

phase.add_state('w', fix_initial=True, fix_final=False, units='m/s', rate_source='wdot',
                lower = -1.0, upper = 1.0, ref0 = -0.2, ref = 0.2)

phase.add_state('phi', fix_initial=True, fix_final=False, units='rad', rate_source='phidot',
                lower = -1/3*np.pi, upper = 1/3*np.pi, ref0 = -1/6*np.pi, ref = 1/6*np.pi)

phase.add_state('theta', fix_initial=True, fix_final=False, units='rad', rate_source='thetadot',
                lower = -1/2*np.pi, upper = 1/2*np.pi, ref0 = -1/6*np.pi, ref = 1/6*np.pi)

phase.add_state('psi', fix_initial=True, fix_final=True, units='rad', rate_source='psidot',
                lower = -1/2*np.pi, upper = 1/2*np.pi, ref0 = -1/3*np.pi, ref = 1/3*np.pi)

phase.add_state('p', fix_initial=True, fix_final=False, units='rad/s', rate_source='pdot',
                lower = -1/3*np.pi/2, upper = 1/3*np.pi/2, ref0 = -1/6*np.pi/2, ref = 1/6*np.pi/2)

phase.add_state('q', fix_initial=True, fix_final=False, units='rad/s', rate_source='qdot',
                lower = -1/2*np.pi/2, upper = 1/2*np.pi/2, ref0 = -1/3*np.pi/2, ref = 1/3*np.pi/2)

phase.add_state('r', fix_initial=True, fix_final=False, units='rad/s', rate_source='rdot',
                lower = -1/2*np.pi/2, upper = 1/2*np.pi/2, ref0 = -1/3*np.pi/2, ref = 1/3*np.pi/2)

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
phase.add_control('n1', units = '1/s', opt=True, lower = 200.0, upper = 400.0,
                   rate_continuity=True)
phase.add_control('n2', units = '1/s', opt=True, lower = 200.0, upper = 400.0,
                   rate_continuity=True)
phase.add_control('n3', units = '1/s', opt=True, lower = 200.0, upper = 400.0,
                   rate_continuity=True)
phase.add_control('n4', units = '1/s', opt=True, lower = 200.0, upper = 400.0,
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

'''minimum time'''
# phase.add_objective('time',  loc='final', ref=1.0)

'''minimum energy/ mechanical power'''
phase.add_objective('J', loc = 'final', ref = 1.0)

# phase.add_boundary_constraint('x', loc='final', equals=9.9932)
# phase.add_boundary_constraint('y', loc='final', equals=-0.2189)
# phase.add_boundary_constraint('z', loc='final', equals=-1.7839)
# phase.add_boundary_constraint('psi', loc='final', equals=0.3609)
# phase.add_boundary_constraint('F_z', loc = 'initial', equals = 6.762)
# phase.add_boundary_constraint('tau_x', loc = 'initial', equals = 0.0)
# phase.add_boundary_constraint('tau_y', loc = 'initial', equals = 0.0)
# phase.add_boundary_constraint('tau_z', loc = 'initial', equals = 0.0)

phase.add_boundary_constraint('n1', loc = 'initial', equals = 250.0)
phase.add_boundary_constraint('n2', loc = 'initial', equals = 250.0)
phase.add_boundary_constraint('n3', loc = 'initial', equals = 250.0)
phase.add_boundary_constraint('n4', loc = 'initial', equals = 250.0)

# p.model.linear_solver = om.DirectSolver()

p.setup(check=True)



p.set_val('traj.phase0.t_initial', 0.0, units='s')
p.set_val('traj.phase0.t_duration', 10.0, units='s')


p.set_val('traj.phase0.states:x', phase.interp('x', [0.0, 9.9688]), units = 'm')
p.set_val('traj.phase0.states:y', phase.interp('y', [0.0, -0.6902]), units = 'm')
p.set_val('traj.phase0.states:z', phase.interp('z', [-2.0, -1.733]), units = 'm')

p.set_val('traj.phase0.states:u', phase.interp('u', [0, 0]), units = 'm/s')
p.set_val('traj.phase0.states:v', phase.interp('v', [0, 0]), units = 'm/s')
p.set_val('traj.phase0.states:w', phase.interp('w', [0, 0]), units = 'm/s')
p.set_val('traj.phase0.states:phi', phase.interp('phi', [0.0, 0.0]), units = 'rad')
p.set_val('traj.phase0.states:theta', phase.interp('theta', [0.0, 0.0]), units = 'rad')
p.set_val('traj.phase0.states:psi', phase.interp('psi', [0.0, -0.3156]), units = 'rad')
p.set_val('traj.phase0.states:p', phase.interp('p', [0.0, 0.0]), units = 'rad/s')
p.set_val('traj.phase0.states:q', phase.interp('q', [0.0, 0.0]), units = 'rad/s')
p.set_val('traj.phase0.states:r', phase.interp('r', [0.0, 0.0]), units = 'rad/s')
p.set_val('traj.phase0.states:J', phase.interp('J', ys=[0, 1]), units = '1/s**2')
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

p.set_val('traj.phase0.controls:n1',phase.interp('n1', ys=[250.0, 250.0]), units='1/s') 
p.set_val('traj.phase0.controls:n2',phase.interp('n2', ys=[250.0, 250.0]), units='1/s') 
p.set_val('traj.phase0.controls:n3',phase.interp('n3', ys=[250.0, 250.0]), units='1/s') 
p.set_val('traj.phase0.controls:n4',phase.interp('n4', ys=[250.0, 250.0]), units='1/s') 


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
t_sol = p.get_val('traj.phase0.timeseries.time')
J = p.get_val('traj.phase0.timeseries.states:J')
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

t_sim = sim.get_val('traj.phase0.timeseries.time')
# F_z = sim.get_val('traj.phase0.timeseries.controls:F_z')
# tau_x = sim.get_val('traj.phase0.timeseries.controls:tau_x')
# tau_y = sim.get_val('traj.phase0.timeseries.controls:tau_y')
# tau_z = sim.get_val('traj.phase0.timeseries.controls:tau_z')
# input_table = np.hstack((np.hstack((np.hstack((np.hstack((t_sim, F_z)), tau_x)), tau_y)), tau_z))

# from scipy.io import savemat
# mdic_in = {"input_table": input_table, "label": "input_table"}
# savemat("input_table.mat", mdic_in)


x = sim.get_val('traj.phase0.timeseries.states:x')
y = sim.get_val('traj.phase0.timeseries.states:y')
z = sim.get_val('traj.phase0.timeseries.states:z')
psi = sim.get_val('traj.phase0.timeseries.states:psi')

output_table = np.hstack((np.hstack((np.hstack((np.hstack((t_sim, x)), y)), z)), psi))
xline, yline, zline = output_table[:,1], output_table[:,2],-output_table[:,3]
# xline, yline, zline = x_array.flatten(), y_array.flatten(), -z_array.flatten()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(xline, yline, zline, 'gray')
# mdic_out = {"output_table": output_table, "label": "output_table"}
# savemat("output_table.mat", mdic_out)
                         
plot_results([('traj.phase0.timeseries.time', 'traj.phase0.timeseries.states:x',
                'time (s)', 'x (m)'),
              ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.states:y',
                'time (s)', 'y (m)'),
              ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.states:z',
                'time (s)', 'z (m)'),
              ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.states:psi',
                'time (s)', 'psi (rad)'),
              ], 
             
              title='Quadrotor outputs', 
             
              p_sol=sol, p_sim = sim)

plt.show()

plot_results([('traj.phase0.timeseries.time', 'traj.phase0.timeseries.controls:n1',
                               'time (s)', 'n1 (revs/s)'),
              ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.controls:n2',
                              'time (s)', 'n2 (revs/s)'),
              ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.controls:n3',
                              'time (s)', 'n3 (revs/s)'),              
              ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.controls:n4',
                               'time (s)', 'n4 (revs/s)')],
              title='Quadrotor control inputs', 
              p_sol=sol, p_sim = sim)

plt.show()

plot_results([('traj.phase0.timeseries.time', 'traj.phase0.timeseries.states:u',
  'time (s)', 'V_x (m/s)'),
              ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.states:v',
                              'time (s)', 'V_y (m/s)'),
              ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.states:w',
                              'time (s)', 'V_z (m/s)')],              
              title='Quadrotor velocity states', 
              p_sol=sol, p_sim = sim)

plt.show()


plot_results([('traj.phase0.timeseries.time', 'traj.phase0.timeseries.states:psi',
  'time (s)', 'yaw (rad)'),
              ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.states:theta',
                              'time (s)', 'pitch (rad)'),
              ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.states:phi',
                              'time (s)', 'roll (rad)')],              
              title='Quadrotor attitude states', 
              p_sol=sol, p_sim = sim)

plt.show()


