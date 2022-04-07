import openmdao.api as om
import dymos as dm
import numpy as np

from quadrotorODE import QuadrotorODE
from dymos.examples.plotting import plot_results
import matplotlib.pyplot as plt

# Instantiate the problem, add the driver, and allow it to use coloring
p = om.Problem(model=om.Group())

p.driver = om.ScipyOptimizeDriver()
#p.driver = om.pyOptSparseDriver()
p.driver.options['optimizer'] = 'SLSQP'
p.driver.declare_coloring()

# p.driver = om.pyOptSparseDriver()
# p.driver.declare_coloring()


# Instantiate a Dymos Trajectory and add it to the Problem model.
# Instantiate the trajectory and phase
traj = dm.Trajectory()

# phase = dm.Phase(ode_class=QuadrotorODE,
#                   transcription=dm.Radau(num_segments=15, order=3))
phase = dm.Phase(ode_class=QuadrotorODE,
                  transcription=dm.GaussLobatto(num_segments=15, compressed=False))
traj.add_phase('phase0', phase)

p.model.add_subsystem('traj', traj)

# Instantiate the trajectory and add a phase to it
# phase0 = traj.add_phase('phase0',
#                         dm.Phase(ode_class=QuadrotorODE,
#                                  transcription=dm.GaussLobatto(num_segments=15, compressed=False)))

phase.set_time_options(fix_initial=True, units='s', duration_bounds=(.05, 20), duration_ref=1)

phase.add_state('x', fix_initial=True, fix_final=True, units='m', rate_source='xdot',
                lower=-1, upper= 11, ref0= -1, ref = 11)

phase.add_state('y', fix_initial=True, fix_final=True, units='m', rate_source='ydot',
                lower=-1, upper= 1, ref0= -1, ref = 1)

phase.add_state('z', fix_initial=True, fix_final=True, units='m', rate_source='zdot',
                lower=-2, upper= -1.5, ref0= -2, ref = -1.5)

phase.add_state('u', fix_initial=True, fix_final=False, units='m/s', rate_source='udot')

phase.add_state('v', fix_initial=True, fix_final=False, units='m/s', rate_source='vdot')

phase.add_state('w', fix_initial=True, fix_final=False, units='m/s', rate_source='wdot')

phase.add_state('phi', fix_initial=True, fix_final=False, units='rad', rate_source='phidot',
                 lower=-1/3. * np.pi, upper= 1/3. * np.pi, ref0= -1/3. * np.pi, ref = 1/3. * np.pi)

phase.add_state('theta', fix_initial=True, fix_final=False, units='rad', rate_source='thetadot',
                 lower=-1/3. * np.pi, upper= 1/3. * np.pi, ref0= -1/3. * np.pi, ref = 1/3. * np.pi)

phase.add_state('psi', fix_initial=True, fix_final=True, units='rad', rate_source='psidot',
                 lower=-1/2. * np.pi, upper= 1/2. * np.pi, ref0= -1/2. * np.pi, ref = 1/2. * np.pi )

phase.add_state('p', fix_initial=True, fix_final=False, units='rad/s', rate_source='pdot')

phase.add_state('q', fix_initial=True, fix_final=False, units='rad/s', rate_source='qdot')

phase.add_state('r', fix_initial=True, fix_final=False, units='rad/s', rate_source='rdot')

phase.add_control('n1', units = '1/s', opt=True, lower = 100, upper = 400)

phase.add_control('n2', units = '1/s', opt=True, lower = 100, upper = 400)

phase.add_control('n3', units = '1/s', opt=True, lower = 100, upper = 400)

phase.add_control('n4', units = '1/s', opt=True, lower = 100, upper = 400)

# phase.add_control('F_z', units = 'N', opt=True, lower = -10, upper = 6.5)
# phase.add_control('tau_x', units = 'N*m', opt=True, lower =-1, upper = 0.2)
# phase.add_control('tau_y', units = 'N*m', opt=True, lower =-6, upper = 2)
# phase.add_control('tau_z', units = 'N*m', opt=True, lower =-0.5, upper = 0.8)
# add outputs
phase.add_timeseries_output('x', shape=(1,))
phase.add_timeseries_output('y', shape=(1,))
phase.add_timeseries_output('z', shape=(1,))
phase.add_timeseries_output('psi', shape=(1,))

phase.add_objective('time',  loc='final', ref=1.0)

#
# Setup the boundary and path constraints
#
# phase.add_boundary_constraint('x', loc='final', equals=9.9932)
# phase.add_boundary_constraint('y', loc='final', equals=-0.2189)
# phase.add_boundary_constraint('z', loc='final', equals=-1.7839)
# phase.add_boundary_constraint('psi', loc='final', equals=0.3609)

# phase.add_path_constraint(name='phi', lower=-1/6. * np.pi, upper= 1/6. * np.pi, ref= 1/6. * np.pi, ref0 = -1/6. * np.pi)
# phase.add_path_constraint(name='theta', lower=-1/6. * np.pi, upper= 1/6. * np.pi, ref= 1/6. * np.pi, ref0 = -1/6. * np.pi)


#p.model.linear_solver = om.DirectSolver()

p.setup(check=True)



p.set_val('traj.phase0.t_initial', 0, units='s')
p.set_val('traj.phase0.t_duration', 5, units='s')


p.set_val('traj.phase0.states:x', phase.interp('x', [0, 9.9932]), units = 'm')
p.set_val('traj.phase0.states:y', phase.interp('y', [0, -0.2189]), units = 'm')
p.set_val('traj.phase0.states:z', phase.interp('z', [-2, -1.7839]), units = 'm')
p.set_val('traj.phase0.states:u', phase.interp('u', [0, 1.5]), units = 'm/s')
p.set_val('traj.phase0.states:v', phase.interp('v', [0, -0.5]), units = 'm/s')
p.set_val('traj.phase0.states:w', phase.interp('w', [0, 0.5]), units = 'm/s')
p.set_val('traj.phase0.states:phi', phase.interp('phi', [0, 0.2]), units = 'rad')
p.set_val('traj.phase0.states:theta', phase.interp('theta', [0, 0.2]), units = 'rad')
p.set_val('traj.phase0.states:psi', phase.interp('psi', [0, 0.3609]), units = 'rad')
p.set_val('traj.phase0.states:p', phase.interp('p', [0, 0]), units = 'rad/s')
p.set_val('traj.phase0.states:q', phase.interp('q', [0, 0]), units = 'rad/s')
p.set_val('traj.phase0.states:r', phase.interp('r', [0, 0]), units = 'rad/s')

# p.set_val('traj.phase0.controls:F_z', phase.interp('F_z', ys=[-6.76, 0]), units='N')
# p.set_val('traj.phase0.controls:tau_x', phase.interp('tau_x', ys=[0, -1]), units='N*m')
# p.set_val('traj.phase0.controls:tau_y', phase.interp('tau_y', ys=[0, -6]), units='N*m')
# p.set_val('traj.phase0.controls:tau_z', phase.interp('tau_z', ys=[0, 0.8]), units='N*m')

p.set_val('traj.phase0.controls:n1', phase.interp('n1', ys=[250, 250]), units='1/s')
p.set_val('traj.phase0.controls:n2', phase.interp('n2', ys=[250, 250]), units='1/s')
p.set_val('traj.phase0.controls:n3', phase.interp('n3', ys=[250, 250]), units='1/s')
p.set_val('traj.phase0.controls:n4', phase.interp('n4', ys=[250, 250]), units='1/s')

# Run the driver
dm.run_problem(p, simulate=True)


sol = om.CaseReader('dymos_solution.db').get_case('final')
sim = om.CaseReader('dymos_simulation.db').get_case('final')

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
             
              p_sol=sol)

plt.show()

plot_results([('traj.phase0.timeseries.time', 'traj.phase0.timeseries.controls:n1',
  'time (s)', 'n1 (revs/s)'),
              ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.states:phi',
                              'time (s)', 'roll (rad)'),
              ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.states:theta',
                              'time (s)', 'pitch (rad)'),              
              ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.states:u',
              'time (s)', 'xdot (m/s)'),
              ('traj.phase0.timeseries.time', 'traj.phase0.timeseries.states:v',
                'time (s)', 'ydot (m/s)')],
              title='Quadrotor input n1 and states', 
              p_sol=sol
              )

plt.show()

