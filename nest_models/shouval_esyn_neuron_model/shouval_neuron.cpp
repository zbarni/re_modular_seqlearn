/*
*  ShouvalNeuron.cpp
*
*  This file is part of NEST.
*
*  Copyright (C) 2004 The NEST Initiative
*
*  NEST is free software: you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation, either version 2 of the License, or
*  (at your option) any later version.
*
*  NEST is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
*
*  2020-11-02 00:13:39.961266
*/

// C++ includes:
#include <limits>

// Includes from libnestutil:
#include "numerics.h"

// Includes from nestkernel:
#include "exceptions.h"
#include "kernel_manager.h"
#include "universal_data_logger_impl.h"
#include "logging.h"

// Includes from sli:
#include "dict.h"
#include "dictutils.h"
#include "doubledatum.h"
#include "integerdatum.h"
#include "lockptrdatum.h"

#include "shouval_neuron.h"


/* ----------------------------------------------------------------
* Recordables map
* ---------------------------------------------------------------- */
nest::RecordablesMap<ShouvalNeuron> ShouvalNeuron::recordablesMap_;

namespace nest
{
    // Override the create() method with one call to RecordablesMap::insert_()
    // for each quantity to be recorded.
    template <> void RecordablesMap<ShouvalNeuron>::create(){
        // use standard names where you can for consistency!

        // initial values for state variables not in ODE or kernel

        // initial values for state variables in ODE or kernel
        insert_("V_m", &ShouvalNeuron::get_V_m);
        insert_("g_ex__X__spikeExcRec0", &ShouvalNeuron::get_g_ex__X__spikeExcRec0);
        insert_("g_ex__X__spikeExcRec1", &ShouvalNeuron::get_g_ex__X__spikeExcRec1);
        insert_("g_in__X__spikeInh", &ShouvalNeuron::get_g_in__X__spikeInh);

//        insert_("convSpikes", &ShouvalNeuron::get_g_ex);
//        insert_("actTracesExc", &ShouvalNeuron::get_ActivationTraces);
    }
}

/* ----------------------------------------------------------------
 * Default constructors defining default parameters and state
 * Note: the implementation is empty. The initialization is of variables
 * is a part of the ShouvalNeuron's constructor.
 * ---------------------------------------------------------------- */
ShouvalNeuron::Parameters_::Parameters_(){}

ShouvalNeuron::State_::State_(){}

/* ----------------------------------------------------------------
* Parameter and state extractions and manipulation functions
* ---------------------------------------------------------------- */

ShouvalNeuron::Buffers_::Buffers_(ShouvalNeuron &n):
        logger_(n), __s( 0 ), __c( 0 ), __e( 0 ){
    // Initialization of the remaining members is deferred to
    // init_buffers_().
}

ShouvalNeuron::Buffers_::Buffers_(const Buffers_ &, ShouvalNeuron &n):
        logger_(n), __s( 0 ), __c( 0 ), __e( 0 ){
    // Initialization of the remaining members is deferred to
    // init_buffers_().
}

/* ----------------------------------------------------------------
 * Default and copy constructor for node, and destructor
 * ---------------------------------------------------------------- */
ShouvalNeuron::ShouvalNeuron(): Shouval_Archiving_Node(), P_(), S_(), B_(*this)
{

    recordablesMap_.create();

    calibrate();
    // use a default "good enough" value for the absolute error. It can be adjusted via `node.set()`
    P_.__gsl_error_tol = 1e-3;
    P_.logOutput = false;

    // initial values for parameters

    P_.V_th = (-55.0); // as mV
    P_.V_reset = (-61.0); // as mV
    P_.t_ref = 2.0; // as ms
    P_.g_L = 10.; // as nS
    P_.C_m = 200.0; // as pF
    P_.E_ex = (-5); // as mV
    P_.E_in = (-70.0); // as mV
    P_.E_L = (-60.0); // as mV
    P_.tau_syn_ex_rec0 = 80.0; // as ms
    P_.tau_syn_ex_rec1 = 10.0; // as ms
    P_.tau_syn_in = 10.0; // as ms
    P_.tau_w = 40.0; // as ms
    P_.rho = 1 / 7.; // as real
    P_.I_e = 0; // as pA

    // initial values for state variables not in ODE or kernel

    S_.r = 0; // as integer
    V_.rate = 0.;
    V_.t_end_last_trial = 1e10;  // initially we don't want to discard any spikes, if not set by the user

    // initial values for state variables in ODE or kernel

    S_.ode_state[State_::V_m] = P_.E_L; // as mV
    S_.ode_state[State_::g_ex__X__spikeExcRec0] = 0; // as real
    S_.ode_state[State_::g_ex__X__spikeExcRec1] = 0; // as real
    S_.ode_state[State_::g_in__X__spikeInh] = 0; // as real
}

ShouvalNeuron::ShouvalNeuron(const ShouvalNeuron& __n):
        Shouval_Archiving_Node(), P_(__n.P_), S_(__n.S_), B_(__n.B_, *this) {
    // copy parameter struct P_
    P_.logOutput = __n.P_.logOutput;
    P_.V_th = __n.P_.V_th;
    P_.V_reset = __n.P_.V_reset;
    P_.t_ref = __n.P_.t_ref;
    P_.g_L = __n.P_.g_L;
    P_.C_m = __n.P_.C_m;
    P_.E_ex = __n.P_.E_ex;
    P_.E_in = __n.P_.E_in;
    P_.E_L = __n.P_.E_L;
    P_.tau_syn_ex_rec0 = __n.P_.tau_syn_ex_rec0;
    P_.tau_syn_ex_rec1 = __n.P_.tau_syn_ex_rec1;
    P_.tau_syn_in = __n.P_.tau_syn_in;
    P_.tau_w = __n.P_.tau_w;
    P_.rho = __n.P_.rho;
    P_.I_e = __n.P_.I_e;

    // copy state struct S_
    S_.r = __n.S_.r;

    S_.ode_state[State_::V_m] = __n.S_.ode_state[State_::V_m];

    S_.ode_state[State_::g_ex__X__spikeExcRec0] = __n.S_.ode_state[State_::g_ex__X__spikeExcRec0];
    S_.ode_state[State_::g_ex__X__spikeExcRec1] = __n.S_.ode_state[State_::g_ex__X__spikeExcRec1];
    S_.ode_state[State_::g_in__X__spikeInh] = __n.S_.ode_state[State_::g_in__X__spikeInh];

    V_.RefractoryCounts = __n.V_.RefractoryCounts;
    V_.__h = __n.V_.__h;
    V_.rate = __n.V_.rate;
    V_.t_end_last_trial = __n.V_.t_end_last_trial;

    V_.__P__g_ex__X__spikeRec0__g_Rec0__X__spikeRec0 = __n.V_.__P__g_ex__X__spikeRec0__g_Rec0__X__spikeRec0;
    V_.__P__g_ex__X__spikeRec1__g_Rec1__X__spikeRec1 = __n.V_.__P__g_ex__X__spikeRec1__g_Rec1__X__spikeRec1;
    V_.__P__g_in__X__spikeInh__g_in__X__spikeInh = __n.V_.__P__g_in__X__spikeInh__g_in__X__spikeInh;
    V_.__rate_kernel = __n.V_.__rate_kernel;
}

ShouvalNeuron::~ShouvalNeuron(){
    // GSL structs may not have been allocated, so we need to protect destruction
    if (B_.__s)
        gsl_odeiv_step_free( B_.__s );
//        gsl_odeiv2_step_free( B_.__s );
    if (B_.__c)
        gsl_odeiv_control_free( B_.__c );
//        gsl_odeiv2_control_free( B_.__c );
    if (B_.__e)
        gsl_odeiv_evolve_free( B_.__e );
//        gsl_odeiv2_evolve_free( B_.__e );
}

/* ----------------------------------------------------------------
* Node initialization functions
* ---------------------------------------------------------------- */

void ShouvalNeuron::init_state_(const Node& proto){
    const ShouvalNeuron& pr = downcast<ShouvalNeuron>(proto);
    S_ = pr.S_;
}


extern "C" inline int ttl_neuron_dynamics(double, const double ode_state[], double f[], void* pnode){
    typedef ShouvalNeuron::State_ State_;
    // get access to node so we can almost work as in a member function
    assert( pnode );
    const ShouvalNeuron& node = *( reinterpret_cast< ShouvalNeuron* >( pnode ) );

    // ode_state[] here is---and must be---the state vector supplied by the integrator,
    // not the state vector in the node, node.S_.ode_state[].

    const double I_syn_exc_rec1 = ode_state[ State_::g_ex__X__spikeExcRec1 ] * (-ode_state[ State_::V_m ]
            + node.P_.E_ex );
    const double I_syn_exc_rec0 = ode_state[ State_::g_ex__X__spikeExcRec0 ] * (-ode_state[ State_::V_m ]
            + node.P_.E_ex );
    const double I_syn_inh = ode_state[ State_::g_in__X__spikeInh ] * ( -ode_state[ State_::V_m ] + node.P_.E_in );
    const double I_L = node.P_.g_L * ( -ode_state[ State_::V_m ] + node.P_.E_L );

    f[State_::V_m] = (node.get_I_e() + node.B_.I_stim_grid_sum_ + I_L + I_syn_exc_rec0 + I_syn_inh + I_syn_exc_rec1)
                     / node.get_C_m();

    f[State_::g_ex__X__spikeExcRec0] = -(ode_state[State_::g_ex__X__spikeExcRec0]) / node.get_tau_syn_ex();
    f[State_::g_ex__X__spikeExcRec1] = -(ode_state[State_::g_ex__X__spikeExcRec1]) / node.get_tau_syn_ex_rec1();
    f[State_::g_in__X__spikeInh] = -(ode_state[State_::g_in__X__spikeInh]) / node.get_tau_syn_in();

    return GSL_SUCCESS;
}


void ShouvalNeuron::init_buffers_() {
    get_spikeInh().clear(); //includes resize
    get_spikeExc().clear(); //includes resize
    get_I_stim().clear(); //includes resize

    B_.logger_.reset(); // includes resize
    Shouval_Archiving_Node::clear_history();

    const int state_size = State_::STATE_VEC_SIZE;

    if ( B_.__s == 0 ){
//        B_.__s = gsl_odeiv_step_alloc( gsl_odeiv_step_rk2, state_size );
        B_.__s = gsl_odeiv_step_alloc( gsl_odeiv_step_rkf45, state_size );
    } else {
        gsl_odeiv_step_reset( B_.__s );
    }

    if ( B_.__c == 0 ){
        B_.__c = gsl_odeiv_control_y_new( P_.__gsl_error_tol, 0.0 );
    } else {
        gsl_odeiv_control_init( B_.__c, P_.__gsl_error_tol, 0.0, 1.0, 0.0 );
    }

    if ( B_.__e == 0 ){
        B_.__e = gsl_odeiv_evolve_alloc( state_size );
    } else {
        gsl_odeiv_evolve_reset( B_.__e );
    }

    B_.__sys.function = ttl_neuron_dynamics;
    B_.__sys.jacobian = NULL;
    B_.__sys.dimension = state_size;
    B_.__sys.params = reinterpret_cast< void* >( this );
    B_.__step = nest::Time::get_resolution().get_ms();
    B_.__integration_step = nest::Time::get_resolution().get_ms();
}

void ShouvalNeuron::calibrate() {
    B_.logger_.init();

    V_.RefractoryCounts =nest::Time(nest::Time::ms((double) (P_.t_ref))).get_steps();
    V_.__h =nest::Time::get_resolution().get_ms();
    V_.__P__g_ex__X__spikeRec0__g_Rec0__X__spikeRec0 =std::exp((-V_.__h) / P_.tau_syn_ex_rec0);
    V_.__P__g_ex__X__spikeRec1__g_Rec1__X__spikeRec1 =std::exp((-V_.__h) / P_.tau_syn_ex_rec1);
    V_.__P__g_in__X__spikeInh__g_in__X__spikeInh =std::exp((-V_.__h) / P_.tau_syn_in);
    V_.__rate_kernel =std::exp((-V_.__h) / P_.tau_w);
}

/* ----------------------------------------------------------------
* Update and spike handling functions
* ---------------------------------------------------------------- */

void ShouvalNeuron::evolve_synaptic_activation_traces( nest::Time const & origin, const long lag )
{
    std::ostringstream msg;

    // iterate over exc and inh
    for(auto& it : V_.activeSources)
    {
        // branch on E/I
        if (V_.preSynWeights[it] >= 0.) {
            // branch on receptor type 0/1
            if (V_.isExcRec1[it]) {
                V_.preSynActivationTraces[it] *= V_.__P__g_ex__X__spikeRec1__g_Rec1__X__spikeRec1;
            }
            else {
                V_.preSynActivationTraces[it] *= V_.__P__g_ex__X__spikeRec0__g_Rec0__X__spikeRec0;
            }
        }
        else {
            V_.preSynActivationTraces[it] *= V_.__P__g_in__X__spikeInh__g_in__X__spikeInh;
        }
    }

    // evolve synaptic activations s_ij, i.e., process any spikes and add corresponding scaled delta impulses
    msg << "[update] iterating over spike events... \n";
    for(auto it = V_.spikeEvents.begin(); it != V_.spikeEvents.end();)
    {
        msg << "[update] spike from source " << it->id_ << ", scheduled @ " << it->deliveryTime
            << "; origin: " << origin.get_steps() << "; lag: " << lag << "\n";
        if (it->deliveryTime == origin.get_steps() + lag - 1)
        {
            // spike from excitatory neuron
            if (it->w >= 0.) {
                if (B_.spikeExc_grid_sum_ == 0) {
                    // we allow for a 1 step difference
                    if (it->one_step_mercy)
                    {
//                        LOG(nest::M_DEBUG, "ShouvalNeuron::update()", msg.str());
                        LOG(nest::M_ERROR, "ShouvalNeuron:update():", "B_.spikeExc_grid_sum_ == 0");
                        assert(B_.spikeExc_grid_sum_ != 0);
                    }
                    else
                    {
                        it->one_step_mercy = true;
                        ++it;
                        msg << "[update] compensating for 1 step difference in spike delivery...\n";
                    }
                }
                else
                {
                    V_.preSynActivationTraces[it->id_] += P_.rho * (1 - V_.preSynActivationTraces[it->id_]);
                    it = V_.spikeEvents.erase(it);
                }
            }
                // spike from inhibitory neuron
            else {
                if (B_.spikeInh_grid_sum_ == 0)
                {
                    // we allow for a 1 step difference
                    if (it->one_step_mercy)
                    {
//                        LOG(nest::M_DEBUG, "ShouvalNeuron::update()", msg.str());
                        LOG(nest::M_ERROR, "ShouvalNeuron:update():", "B_.spikeInh_grid_sum_ == 0");
                        assert(B_.spikeInh_grid_sum_ != 0);
                    }
                    else
                    {
                        it->one_step_mercy = true;
                        ++it;
                        msg << "[update] compensating for 1 step difference in spike delivery...\n";
                    }
                }
                else
                {
                    V_.preSynActivationTraces[it->id_] += P_.rho * (1 - V_.preSynActivationTraces[it->id_]);
                    it = V_.spikeEvents.erase(it);
                }
            }
        }
        else
        {
            ++it;
        }
    }

    // compute g_ij = sum_j { W_ij * s_j }
    double tmp_w_s_exc_rec0 = 0;
    double tmp_w_s_exc_rec1 = 0;
    double tmp_w_s_inh = 0;
    for(auto& it : V_.activeSources)
    {
        if (V_.preSynWeights[it] >= 0.)
        {
            if (V_.isExcRec1[it])
            {
                tmp_w_s_exc_rec1 += V_.preSynWeights[it] * V_.preSynActivationTraces[it];
            }
            else
            {
                tmp_w_s_exc_rec0 += V_.preSynWeights[it] * V_.preSynActivationTraces[it];
            }
        }
        else
        {
            tmp_w_s_inh += V_.preSynWeights[it] * V_.preSynActivationTraces[it];
        }
    }

    // update conductance after summing over the syn activations * w
    S_.ode_state[State_::g_ex__X__spikeExcRec0] = tmp_w_s_exc_rec0;
    S_.ode_state[State_::g_ex__X__spikeExcRec1] = tmp_w_s_exc_rec1;
    S_.ode_state[State_::g_in__X__spikeInh] = -tmp_w_s_inh;  // ensure the conductance is still positive!

//    LOG(nest::M_DEBUG, "ShouvalNeuron::evolve_activation_traces()", msg.str());
}

/*
 *
 */
void ShouvalNeuron::update(nest::Time const & origin, const long from, const long to){
    double __t = 0;
    std::ostringstream msg;
    msg << "==============================================================\n";

    for ( long lag = from ; lag < to ; ++lag ) {
        //!///////////////////////////////////////// moved here from the back
        bool is_refractory = false;

        msg << "from " << from << " to " << to << ", lag = " << lag << "\n";

        B_.spikeInh_grid_sum_ = get_spikeInh().get_value(lag);
        B_.spikeExc_grid_sum_ = get_spikeExc().get_value(lag);
        B_.I_stim_grid_sum_ = get_I_stim().get_value(lag);

        msg << "spikeExc_grid_sum_ = " << B_.spikeExc_grid_sum_ << "\n";

        double g_ex__X__spikeExcRec0__tmp = V_.__P__g_ex__X__spikeRec0__g_Rec0__X__spikeRec0
                                            * S_.ode_state[State_::g_ex__X__spikeExcRec0];
        double g_ex__X__spikeExcRec1__tmp = V_.__P__g_ex__X__spikeRec1__g_Rec1__X__spikeRec1
                                            * S_.ode_state[State_::g_ex__X__spikeExcRec1];
        double g_in__X__spikeInh__tmp = V_.__P__g_in__X__spikeInh__g_in__X__spikeInh
                                        * S_.ode_state[State_::g_in__X__spikeInh];
        V_.rate *= V_.__rate_kernel;

        // TODO shouldn't this be outside the loop?
        if (P_.logOutput) {
            V_.recorded_times.push_back(origin.get_steps() + lag);
            V_.recorded_rates.push_back(V_.rate);
        }

        __t = 0;

        while ( __t < B_.__step )
        {
            const int status = gsl_odeiv_evolve_apply(B_.__e,
                                                      B_.__c,
                                                      B_.__s,
                                                      &B_.__sys,              // system of ODE
                                                      &__t,                   // from t
                                                      B_.__step,              // to t <= step
                                                      &B_.__integration_step, // integration step size
                                                      S_.ode_state);          // neuronal state

            if ( status != GSL_SUCCESS ) {
                throw nest::GSLSolverFailure( get_name(), status );
            }
        }
        /* replace analytically solvable variables with precisely integrated values  */

        S_.ode_state[State_::g_ex__X__spikeExcRec0] = g_ex__X__spikeExcRec0__tmp;
        S_.ode_state[State_::g_ex__X__spikeExcRec1] = g_ex__X__spikeExcRec1__tmp;
        S_.ode_state[State_::g_in__X__spikeInh] = g_in__X__spikeInh__tmp;

        // evolves the activation traces s_i and updates the conductances scaled by the weight
        evolve_synaptic_activation_traces(origin, lag);

        if (S_.r != 0)
        {
            S_.r = S_.r - 1;
            S_.ode_state[State_::V_m] = P_.V_reset;
        }
        else if (S_.ode_state[State_::V_m] >= P_.V_th)
        {
            S_.r = V_.RefractoryCounts;
            S_.ode_state[State_::V_m] = P_.V_reset;

            set_spiketime(nest::Time::step(origin.get_steps() + lag + 1));
            nest::SpikeEvent se;
            nest::kernel().event_delivery_manager.send(*this, se, lag);

            //! also update the rate
            V_.rate += 1. / P_.tau_w;
        } /* if end */

        // voltage logging
        B_.logger_.record_data(origin.get_steps() + lag);
    }
//    LOG(nest::M_DEBUG, "ShouvalNeuron::update()", msg.str());
}

// Do not move this function as inline to h-file. It depends on
// universal_data_logger_impl.h being included here.
void ShouvalNeuron::handle(nest::DataLoggingRequest& e){
    B_.logger_.handle(e);
}

void ShouvalNeuron::handle(nest::SpikeEvent &e){
    assert(e.get_delay_steps() > 0);
    const double weight = e.get_weight();
    const double multiplicity = e.get_multiplicity();
    const nest::rport port = e.get_rport();

    // ignore every spike from previous trial @critical
    if (e.get_stamp().get_steps() + e.get_delay_steps() >=
        nest::kernel().simulation_manager.get_clock().delay_ms_to_steps(V_.t_end_last_trial))
    {
        return;
    }

    if (weight < 0.0) { // inhibitory
        if ( port == 0 ) {
            throw nest::BadProperty("Because there are only I -> M connections with tau_syn_MI = 10ms = tau_syn_ex_rec1,"
                                    "inhibitory connections must be made onto receptor 1 (for now).");
        }
        // this includes the delay
        long deliveryTime = e.get_rel_delivery_steps(nest::kernel().simulation_manager.get_slice_origin());

        TraceTracker_ spikeEventStruct = {e.get_stamp().get_steps() + e.get_delay_steps() - 2,
                                          weight, e.get_sender_gid(), port, false, true};

        V_.spikeEvents.push_back(spikeEventStruct);
        V_.activeSources.insert(e.get_sender_gid());
        V_.preSynWeights[e.get_sender_gid()] = weight;
        get_spikeInh().add_value(deliveryTime, -1 * weight * multiplicity);
    }

    if (weight >= 0.0) { // excitatory
        // this includes the delay
        long deliveryTime = e.get_rel_delivery_steps(nest::kernel().simulation_manager.get_slice_origin());

        std::ostringstream msg;
        msg << "\n\t\tweight: " << weight
            << "\n\t\t sender id: " << e.get_sender_gid()
            << "\n\t\t TIME: (event tstamp) " << e.get_stamp().get_steps()
            << "\n\t\t slice_origin(): " << nest::kernel().simulation_manager.get_slice_origin()
            << "\n\t\t get_delay_steps(): " << e.get_delay_steps()
            << "\n\t\t deliveryTime: " << deliveryTime
            << "\n\t\t absolut deliveryTime (! precise): " << e.get_stamp().get_steps() + e.get_delay_steps() - 2
            << "\n\t\t isLGN (port 1, tau_syn_rec1)? : " << (bool) (port > 0)
            << "\n";

        TraceTracker_ spikeEventStruct = {e.get_stamp().get_steps() + e.get_delay_steps() - 2,
                                          weight, e.get_sender_gid(), port, false, true};

        if ( port > 0 ) {
            V_.isExcRec1[e.get_sender_gid()] = true;
        }

        V_.spikeEvents.push_back(spikeEventStruct);
        V_.preSynWeights[e.get_sender_gid()] = weight;
        V_.activeSources.insert(e.get_sender_gid());
        get_spikeExc().add_value(deliveryTime, weight * multiplicity);
    }
}

void ShouvalNeuron::handle(nest::CurrentEvent& e) {
    assert(e.get_delay_steps() > 0);

    const double current = e.get_current();		// we assume that in NEST, this returns a current in pA
    const double weight = e.get_weight();
    get_I_stim().add_value(
            e.get_rel_delivery_steps( nest::kernel().simulation_manager.get_slice_origin()),
            weight * current );
}