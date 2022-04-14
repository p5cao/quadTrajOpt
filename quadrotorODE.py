import numpy as np
import openmdao.api as om
from openmdao.api import Group


class Rotordynamics(om.ExplicitComponent):
    def initialize(self):
       self.options.declare('num_nodes', types=int)

    def setup(self):
       nn = self.options['num_nodes']
       self.add_input('n1', val = np.ones(nn), desc ='rotary speed of 1st rotor', units='1/s')
       self.add_input('n2', val = np.ones(nn), desc ='rotary speed of 2nd rotor', units='1/s')
       self.add_input('n3', val = np.ones(nn), desc ='rotary speed of 3rd rotor', units='1/s')
       self.add_input('n4', val = np.ones(nn), desc ='rotary speed of 4th rotor', units='1/s')
       
       self.add_output('F_z', val = np.ones(nn), desc = 'sum of thrusts produced by 4 rotors',
                       units = 'N')
       
       self.add_output('tau_x', val = np.ones(nn), desc = 'sum of roll moments produced by 4 rotors',
                       units = 'N*m')
    
       self.add_output('tau_y', val = np.ones(nn), desc = 'sum of pitch moments produced by 4 rotors',
                       units = 'N*m')
    
       self.add_output('tau_z', val = np.ones(nn), desc = 'sum of yaw moments produced by 4 rotors',
                       units = 'N*m')
       
       partial_range = np.arange(nn, dtype = int)
       self.declare_partials('F_z', 'n1', rows=partial_range, cols=partial_range)
       self.declare_partials('F_z', 'n2', rows=partial_range, cols=partial_range)
       self.declare_partials('F_z', 'n3', rows=partial_range, cols=partial_range)
       self.declare_partials('F_z', 'n4', rows=partial_range, cols=partial_range)
       
       self.declare_partials('tau_x', 'n1', rows=partial_range, cols=partial_range)
       self.declare_partials('tau_x', 'n2', rows=partial_range, cols=partial_range)
       self.declare_partials('tau_x', 'n3', rows=partial_range, cols=partial_range)
       self.declare_partials('tau_x', 'n4', rows=partial_range, cols=partial_range)
       
       self.declare_partials('tau_y', 'n1', rows=partial_range, cols=partial_range)
       self.declare_partials('tau_y', 'n2', rows=partial_range, cols=partial_range)
       self.declare_partials('tau_y', 'n3', rows=partial_range, cols=partial_range)
       self.declare_partials('tau_y', 'n4', rows=partial_range, cols=partial_range)
       
       self.declare_partials('tau_z', 'n1', rows=partial_range, cols=partial_range)
       self.declare_partials('tau_z', 'n2', rows=partial_range, cols=partial_range)
       self.declare_partials('tau_z', 'n3', rows=partial_range, cols=partial_range)
       self.declare_partials('tau_z', 'n4', rows=partial_range, cols=partial_range)

    def compute(self, inputs, outputs):
       C_t = .0409
       C_roll = 0
       C_pitch = 0
       
       rho = 1.225
       R = 0.0762
       D = 2*R
       l_arm = 0.225
       
       '''need work
        C_P ='''
       k_t =  C_t * rho * D**4
       k_yaw = C_t * rho * D**4 * 5.40e-8/3.65e-6 # empirical from paper
       k_pitch = C_pitch * rho * D**5
       k_roll = C_roll* rho * D**5
       
       n1 = inputs['n1']
       n2 = inputs['n2']
       n3 = inputs['n3']
       n4 = inputs['n4']
       
       outputs['F_z'] = k_t *(n1**2 + n2**2 + n3**2 + n4**2)
       outputs['tau_x'] = (k_t*np.sqrt(1/2)*l_arm+k_roll) * \
           (n1**2 - n2**2 - n3**2 + n4**2)
       outputs['tau_y'] = (k_t*np.sqrt(1/2)*l_arm+k_pitch) * n1**2 + \
       (k_t*np.sqrt(1/2)*l_arm+k_pitch) * n2**2 -  (k_t*np.sqrt(1/2)*l_arm+k_pitch) * n3**2 \
           -(k_t*np.sqrt(1/2)*l_arm+k_pitch) * n4**2
       outputs['tau_z'] = k_yaw * n1**2 - k_yaw * n2**2 + k_yaw * n3**2 - k_yaw * n4**2
       
    def compute_partials(self, inputs, J):
       C_t = .0409
       C_roll = 0
       C_pitch = 0
       
       rho = 1.225
       R = 0.0762
       D = 2*R
       l_arm = 0.225
       
       '''need work
        C_P ='''
       k_t =  C_t * rho * D**4
       k_yaw = C_t * rho * D**4 * 5.40e-8/3.65e-6 # empirical from paper
       k_pitch = C_pitch * rho * D**5
       k_roll = C_roll* rho * D**5
       
       n1 = inputs['n1']
       n2 = inputs['n2']
       n3 = inputs['n3']
       n4 = inputs['n4']
       
       J['F_z', 'n1'] = 2 * k_t * n1
       J['F_z', 'n2'] = 2 * k_t * n2
       J['F_z', 'n3'] = 2 * k_t * n3
       J['F_z', 'n4'] = 2 * k_t * n4
       
       J['tau_x','n1'] = 2*(k_t*np.sqrt(1/2)*l_arm+k_roll) * n1
       J['tau_x','n2'] = -2*(k_t*np.sqrt(1/2)*l_arm+k_roll) * n2
       J['tau_x','n3'] = -2*(k_t*np.sqrt(1/2)*l_arm+k_roll) * n3
       J['tau_x','n4'] = 2*(k_t*np.sqrt(1/2)*l_arm+k_roll) * n4
       
       J['tau_y','n1'] = 2*(k_t*np.sqrt(1/2)*l_arm+k_pitch) * n1
       J['tau_y','n2'] = 2*(k_t*np.sqrt(1/2)*l_arm+k_pitch) * n2
       J['tau_y','n3'] = -2*(k_t*np.sqrt(1/2)*l_arm+k_pitch) * n3
       J['tau_y','n4'] = -2*(k_t*np.sqrt(1/2)*l_arm+k_pitch) * n4
       
       J['tau_z', 'n1'] = 2 * k_yaw * n1
       J['tau_z', 'n2'] = -2 * k_yaw * n2
       J['tau_z', 'n3'] = 2 * k_yaw * n3
       J['tau_z', 'n4'] = -2 * k_yaw * n4
       
       
class Flightdynamics(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        
    def setup(self):
        nn = self.options['num_nodes']
        # states
        self.add_input('phi', val = np.ones(nn), desc='roll angle in world frame', units = 'rad' )
        self.add_input('theta', val = np.ones(nn), desc='pitch angle in world frame', units = 'rad' )
        self.add_input('psi', val = np.ones(nn), desc='yaw angle in world frame', units = 'rad' )
        self.add_input('p', val = np.ones(nn),  desc='roll rate in body frame', units = 'rad/s')
        self.add_input('q', val = np.ones(nn),  desc='pitch rate in body frame', units = 'rad/s')
        self.add_input('r', val = np.ones(nn),  desc='yaw rate in body frame', units = 'rad/s')
        self.add_input('x', val = np.ones(nn), desc='x coordinate in world frame', units = 'm' )
        self.add_input('y', val = np.ones(nn), desc='y coordinate in world frame', units = 'm' )
        self.add_input('z', val = np.ones(nn), desc='z coordinate in world frame', units = 'm' )
        self.add_input('u', val = np.ones(nn), desc='x vel comp in world frame', units = 'm/s' )
        self.add_input('v', val = np.ones(nn), desc='y vel comp in world frame', units = 'm/s' )
        self.add_input('w', val = np.ones(nn), desc='x vel comp in world frame', units = 'm/s' )
        
        # controls
        self.add_input('F_z', val = np.ones(nn), desc = 'sum of thrusts produced by 4 rotors',
                        units = 'N')
        
        self.add_input('tau_x', val = np.ones(nn), desc = 'sum of roll moments produced by 4 rotors',
                        units = 'N*m')
        
        self.add_input('tau_y', val = np.ones(nn), desc = 'sum of pitch moments produced by 4 rotors',
                        units = 'N*m')
     
        self.add_input('tau_z', val = np.ones(nn), desc = 'sum of yaw moments produced by 4 rotors',
                        units = 'N*m')
        
        # left hand side
        self.add_output('xdot', val = np.ones(nn), desc = 'x vel comp in world frame', units = 'm/s')
        self.add_output('ydot', val = np.ones(nn), desc = 'y vel comp in world frame', units = 'm/s')
        self.add_output('zdot', val = np.ones(nn), desc = 'z vel comp in world frame', units = 'm/s')
        
        self.add_output('udot', val = np.ones(nn), desc = 'x accel comp in world frame', units = 'm/s**2')
        self.add_output('vdot', val = np.ones(nn), desc = 'y accel comp in world frame', units = 'm/s**2')
        self.add_output('wdot', val = np.ones(nn), desc = 'z accel comp in world frame', units = 'm/s**2')
        
        self.add_output('phidot', val = np.ones(nn), desc = 'roll rate in world frame', units = 'rad/s')
        self.add_output('thetadot', val = np.ones(nn), desc = 'pitch rate in world frame', units = 'rad/s')
        self.add_output('psidot', val = np.ones(nn), desc = 'yaw rate in world frame', units = 'rad/s')
        
        self.add_output('pdot', val = np.ones(nn), desc = 'roll accel in body frame', units = 'rad/s**2')
        self.add_output('qdot', val = np.ones(nn), desc = 'pitch accel in body frame', units = 'rad/s**2')
        self.add_output('rdot', val = np.ones(nn), desc = 'yaw accel in body frame', units = 'rad/s**2')
        
        # partials
        partial_range = np.arange(nn, dtype = int)
        
        self.declare_partials('xdot', 'u', rows = partial_range, cols = partial_range)
        self.declare_partials('ydot', 'v', rows = partial_range, cols = partial_range)
        self.declare_partials('zdot', 'w', rows = partial_range, cols = partial_range)
        
        self.declare_partials('udot','F_z', rows = partial_range, cols = partial_range)
        self.declare_partials('udot','phi', rows = partial_range, cols = partial_range)
        self.declare_partials('udot','theta', rows = partial_range, cols = partial_range)
        self.declare_partials('udot','psi', rows = partial_range, cols = partial_range)
        
        self.declare_partials('vdot','F_z', rows = partial_range, cols = partial_range)
        self.declare_partials('vdot','phi', rows = partial_range, cols = partial_range)
        self.declare_partials('vdot','theta', rows = partial_range, cols = partial_range)
        self.declare_partials('vdot','psi', rows = partial_range, cols = partial_range)
        
        self.declare_partials('wdot','F_z', rows = partial_range, cols = partial_range)
        self.declare_partials('wdot','phi', rows = partial_range, cols = partial_range)
        self.declare_partials('wdot','theta', rows = partial_range, cols = partial_range)
        
        self.declare_partials('phidot','phi', rows = partial_range, cols = partial_range)
        self.declare_partials('phidot','theta', rows = partial_range, cols = partial_range)
        self.declare_partials('phidot','p', rows = partial_range, cols = partial_range)
        self.declare_partials('phidot','q', rows = partial_range, cols = partial_range)
        self.declare_partials('phidot','r', rows = partial_range, cols = partial_range)
        
        self.declare_partials('thetadot','phi', rows = partial_range, cols = partial_range)
        self.declare_partials('thetadot','q', rows = partial_range, cols = partial_range)
        self.declare_partials('thetadot','r', rows = partial_range, cols = partial_range)
        
        self.declare_partials('psidot','phi', rows = partial_range, cols = partial_range)
        self.declare_partials('psidot','theta', rows = partial_range, cols = partial_range)
        self.declare_partials('psidot','q', rows = partial_range, cols = partial_range)
        self.declare_partials('psidot','r', rows = partial_range, cols = partial_range)
        
        self.declare_partials('pdot','q', rows = partial_range, cols = partial_range)
        self.declare_partials('pdot','r', rows = partial_range, cols = partial_range)
        self.declare_partials('pdot','tau_x', rows = partial_range, cols = partial_range)
        
        self.declare_partials('qdot','p', rows = partial_range, cols = partial_range)
        self.declare_partials('qdot','r', rows = partial_range, cols = partial_range)
        self.declare_partials('qdot','tau_y', rows = partial_range, cols = partial_range)

        self.declare_partials('rdot','p', rows = partial_range, cols = partial_range)
        self.declare_partials('rdot','q', rows = partial_range, cols = partial_range)
        self.declare_partials('rdot','tau_z', rows = partial_range, cols = partial_range)        
        
        
    def compute(self, inputs, outputs):
        m = 0.69 #kg
        g = 9.80 #m/s^2
        l_arm = 0.225 # m
        config = "4x"
        I_x = 0.0469 #kg m^-2
        I_y = 0.0358 
        I_z = 0.0673
        I_r = 3.357e-5 # rotor mom of inertia
        
        x,y,z = inputs['x'], inputs['y'], inputs['z']
        u,v,w = inputs['u'], inputs['v'], inputs['w']
        p,q,r = inputs['p'], inputs['q'], inputs['r']
        phi, theta, psi = inputs['phi'], inputs['theta'], inputs['psi']
        F_z, tau_x, tau_y, tau_z = inputs['F_z'], inputs['tau_x'], inputs['tau_y'], inputs['tau_z']
        s_phi = np.sin(phi)
        c_phi = np.cos(phi)
        s_theta = np.sin(theta)
        c_theta = np.cos(theta)
        s_psi, c_psi = np.sin(psi), np.cos(psi)
        t_theta = np.tan(theta)
        
        g_7 = -1/m * (s_phi*s_psi + c_phi*c_psi*s_theta)
        g_8 = -1/m * (c_phi*s_psi*s_theta - c_psi*s_phi)
        g_9 = -1/m *(c_phi*c_theta)
        
        outputs['xdot'] = u
        outputs['ydot'] = v
        outputs['zdot'] = w
        outputs['psidot'] = q*s_phi/c_theta + r * c_phi / c_theta
        outputs['thetadot'] = q * c_phi - r * s_phi
        outputs['phidot'] = p + q * s_phi * t_theta + r * c_phi * t_theta
        outputs['udot'] = g_7 * F_z
        outputs['vdot'] = g_8 * F_z
        outputs['wdot'] = g + g_9 * F_z
        outputs['udot'] = g_7 * F_z
        outputs['vdot'] = g_8 * F_z
        outputs['wdot'] = g + g_9 * F_z
        outputs['pdot'] = (I_y - I_z)/I_x * q * r + 1/I_x * tau_x
        outputs['qdot'] = (I_z - I_x)/I_y * p * r + 1/I_y * tau_y
        outputs['rdot'] = (I_x - I_y)/I_z * p * q + 1/I_z * tau_z
        
        
    def compute_partials(self, inputs, J):
        m = 0.69 #kg
        g = 9.80 #m/s^2
        l_arm = 0.225 # m
        config = "4x"
        I_x = 0.0469 #kg m^-2
        I_y = 0.0358 
        I_z = 0.0673
        I_r = 3.357e-5 # rotor mom of inertia
        
        x,y,z = inputs['x'], inputs['y'], inputs['z']
        u,v,w = inputs['u'], inputs['v'], inputs['w']
        p,q,r = inputs['p'], inputs['q'], inputs['r']
        phi, theta, psi = inputs['phi'], inputs['theta'], inputs['psi']
        F_z, tau_x, tau_y, tau_z = inputs['F_z'], inputs['tau_x'], inputs['tau_y'], inputs['tau_z']
        s_phi = np.sin(phi)
        c_phi = np.cos(phi)
        s_theta = np.sin(theta)
        c_theta = np.cos(theta)
        s_psi, c_psi = np.sin(psi), np.cos(psi)
        t_theta = np.tan(theta)
        
        g_7 = -1/m * (s_phi*s_psi + c_phi*c_psi*s_theta)
        g_8 = 1/m * (c_psi*s_phi - c_phi*s_psi*s_theta)
        g_9 = -1/m *(c_phi*c_theta)
        
        dg7_dphi = (c_psi*s_phi*s_theta - c_phi*s_psi)/m
        dg7_dtheta = -(c_phi*c_psi*c_theta)/m
        dg7_dpsi = -(c_psi*s_phi - c_phi*s_psi*s_theta)/m
        
        dg8_dphi = (c_phi * c_psi + s_phi *s_psi * s_theta)/m
        dg8_dtheta = -(c_phi*c_theta*s_psi)/m
        dg8_dpsi = -(s_phi*s_psi + c_phi*c_psi*s_theta)/m
        
        dg9_dphi = (c_theta * s_phi)/m
        dg9_dtheta = (c_phi * s_theta)/m

        J['xdot', 'u'] = 1
        J['ydot', 'v'] = 1
        J['zdot', 'w'] = 1
        
        J['udot', 'F_z'] = g_7
        J['udot', 'phi'] = dg7_dphi * F_z
        J['udot', 'theta'] = dg7_dtheta * F_z
        J['udot', 'psi'] = dg7_dpsi * F_z
        
        J['vdot', 'F_z'] = g_8
        J['vdot', 'phi'] = dg8_dphi * F_z
        J['vdot', 'theta'] = dg8_dtheta * F_z
        J['vdot', 'psi'] = dg8_dpsi * F_z
        
        J['wdot', 'F_z'] = g_9
        J['wdot', 'phi'] = dg9_dphi * F_z
        J['wdot', 'theta'] = dg9_dtheta * F_z
        
        J['phidot','phi'] = q*c_phi*t_theta - r*s_phi*t_theta
        J['phidot','theta'] = r*c_phi*(t_theta**2 + 1) + q*s_phi*(t_theta**2 + 1)
        J['phidot', 'p'] = 1       
        J['phidot', 'q'] = s_phi*t_theta
        J['phidot', 'r'] = c_phi*t_theta
        
        J['thetadot', 'phi'] = - r*c_phi - q*s_phi
        J['thetadot', 'q'] = c_phi
        J['thetadot', 'r'] = -s_phi
        
        J['psidot', 'phi'] = (q*c_phi)/c_theta - (r*s_phi)/c_theta
        J['psidot', 'theta'] = (s_theta*(r*c_phi + q*s_phi))/c_theta**2
        J['psidot', 'q'] = s_phi/c_theta
        J['psidot', 'r'] = c_phi/c_theta
        
        J['pdot', 'q'] = (r*(I_y - I_z))/I_x
        J['pdot', 'r'] = (q*(I_y - I_z))/I_x
        J['pdot','tau_x'] = 1/I_x
        
        J['qdot', 'p'] = -(r*(I_x - I_z))/I_y
        J['qdot', 'r'] = -(p*(I_x - I_z))/I_y
        J['qdot','tau_y'] = 1/I_y
        
        J['rdot','p'] = (I_x - I_y)/I_z * q
        J['rdot','q'] = (I_x - I_y)/I_z * p
        J['rdot','tau_z'] = 1/I_z
        
        
# Assemble the ODE
class QuadrotorODE(Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        
        self.add_subsystem('rotor', subsys = Rotordynamics(num_nodes=nn),
                           promotes_inputs=['n1','n2','n3','n4'],
                           promotes_outputs=['F_z', 'tau_x', 'tau_y', 'tau_z'])
        self.add_subsystem('eom', subsys = Flightdynamics(num_nodes=nn),
                           promotes_inputs=['F_z', 'tau_x', 'tau_y', 'tau_z', 
                                            'x', 'y', 'z', 'u', 'v', 'w',
                                            'p', 'q', 'r', 'phi','theta','psi'],
                           promotes_outputs = ['xdot', 'ydot', 'zdot', 'udot',
                                               'vdot', 'wdot', 'pdot', 'qdot',
                                               'rdot', 'phidot', 'thetadot',
                                               'psidot'])
        