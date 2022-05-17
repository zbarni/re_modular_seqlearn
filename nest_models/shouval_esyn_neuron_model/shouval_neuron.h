
/*
*  ShouvalNeuron.h
*  ===================================================================
*  This version of the Shouval neuron model implements the eligibility
*  traces as part of the neuron and not synapse model.
*  ===================================================================
*
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
*  2020-11-02 00:13:39.823212
*/
#ifndef TTL_NEURON
#define TTL_NEURON

#include "config.h"


#ifdef HAVE_GSL

#include <fstream>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>
//#include <gsl/gsl_odeiv2.h>
#include <sstream>

// forwards the declaration of the function
/**
 * Function computing right-hand side of ODE for GSL solver.
 * @note Must be declared here so we can befriend it in class.
 * @note Must have C-linkage for passing to GSL. Internally, it is
 *       a first-class C++ function, but cannot be a member function
 *       because of the C-linkage.
 * @note No point in declaring it inline, since it is called
 *       through a function pointer.
 * @param void* Pointer to model neuron instance.
 */
extern "C" inline int ttl_neuron_dynamics( double, const double y[], double f[], void* pnode );


// Includes from nestkernel:
//#include "archiving_node.h"
#include "shouval_archiving_node.h"
#include "connection.h"
#include "event.h"
#include "nest_types.h"
#include "ring_buffer.h"
#include "universal_data_logger.h"
#include "logging.h"


// Includes from sli:
#include "dictdatum.h"

/* BeginDocumentation
  Name: ShouvalNeuron.

  Description:  
    



  Description
  +++++++++++

  See also
  ++++++++


  Author
  ++++++




  Parameters:
  The following parameters can be set in the status dictionary.
  V_th [mV]  Threshold Potential
  V_reset [mV]  Reset Potential
  t_ref [ms]  Refractory period
  g_L [nS]  Leak Conductance, 0.001 microS
  C_m [pF]  Membrane Capacitance, 20*g_L in nF
  E_ex [mV]  Excitatory reversal Potential
  E_in [mV]  Inhibitory reversal Potential
  E_L [mV]  Leak reversal Potential (aka resting potential)
  tau_syn_ex_rec0 [ms]  Synaptic Time Constant Excitatory Synapse
  tau_syn_in [ms]  Synaptic Time Constant for Inhibitory Synapse
  rho [real]  Fractional change of synaptic activation
  I_e [pA]  constant external input current
  

  Dynamic state variables:
  r [integer]  counts number of tick during the refractory period
  

  Initial values:
  V_m [mV]  membrane potential
  

  References: Empty

  Sends: nest::SpikeEvent

  Receives: Spike, Current, DataLoggingRequest
*/
class ShouvalNeuron : public nest::Shouval_Archiving_Node{
public:
    /**
    * The constructor is only used to create the model prototype in the model manager.
    */
    ShouvalNeuron();

    /**
    * The copy constructor is used to create model copies and instances of the model.
    * @node The copy constructor needs to initialize the parameters and the state.
    *       Initialization of buffers and interal variables is deferred to
    *       @c init_buffers_() and @c calibrate().
    */
    ShouvalNeuron(const ShouvalNeuron &);

    /**
    * Releases resources.
    */
    ~ShouvalNeuron() override;

    /**
     * Import sets of overloaded virtual functions.
     * @see Technical Issues / Virtual Functions: Overriding, Overloading, and
     * Hiding
     */
    using nest::Node::handles_test_event;
    using nest::Node::handle;

    /**
    * Used to validate that we can send nest::SpikeEvent to desired target:port.
    */
    nest::port send_test_event(nest::Node& target, nest::rport receptor_type, nest::synindex, bool);

    /**
    * @defgroup mynest_handle Functions handling incoming events.
    * We tell nest that we can handle incoming events of various types by
    * defining @c handle() and @c connect_sender() for the given event.
    * @{
    */
    void handle(nest::SpikeEvent &);        //! accept spikes
    void handle(nest::CurrentEvent &);      //! accept input current
    void handle(nest::DataLoggingRequest &);//! allow recording with multimeter

    nest::port handles_test_event(nest::SpikeEvent&, nest::port);
    nest::port handles_test_event(nest::CurrentEvent&, nest::port);
    nest::port handles_test_event(nest::DataLoggingRequest&, nest::port);
    /** @} */

    // SLI communication functions:
    void get_status(DictionaryDatum &) const;
    void set_status(const DictionaryDatum &);

//private:
    //! Reset parameters and state of neuron.

    //! Reset state of neuron.
    void init_state_(const Node& proto);

    //! Reset internal buffers of neuron.
    void init_buffers_();

    //! Initialize auxiliary quantities, leave parameters and state untouched.
    void calibrate();

    //! Take neuron through given time interval
    void update(nest::Time const &, const long, const long);
    void evolve_synaptic_activation_traces( nest::Time const &, const long );

    // The next two classes need to be friends to access the State_ class/member
    friend class nest::RecordablesMap<ShouvalNeuron>;
    friend class nest::UniversalDataLogger<ShouvalNeuron>;

    struct TraceTracker_{
        long deliveryTime;
        double w;
        unsigned long id_;
        nest::rport port;
        // this allows a one simulation step difference in spike delivery, in order to allow correct processing;
        // spike_generator and poisson_generators call the handle() function with a 1 step difference, and we must
        // compensate for this by waiting for 1 extra step for the spike arrival during the update().
        bool one_step_mercy;
        bool is_exc;

    };

    /**
    * Free parameters of the neuron.
    *
    *
    *
    * These are the parameters that can be set by the user through @c `node.set()`.
    * They are initialized from the model prototype when the node is created.
    * Parameters do not change during calls to @c update() and are not reset by
    * @c ResetNetwork.
    *
    * @note Parameters_ need neither copy constructor nor @c operator=(), since
    *       all its members are copied properly by the default copy constructor
    *       and assignment operator. Important:
    *       - If Parameters_ contained @c Time members, you need to define the
    *         assignment operator to recalibrate all members of type @c Time . You
    *         may also want to define the assignment operator.
    *       - If Parameters_ contained members that cannot copy themselves, such
    *         as C-style arrays, you need to define the copy constructor and
    *         assignment operator to copy those members.
    */
    struct Parameters_{

        bool logOutput;

        //!  Threshold Potential
        double V_th;

        //!  Reset Potential
        double V_reset;

        //!  Refractory period
        double t_ref;

        //!  Leak Conductance, 0.001 microS
        double g_L;

        //!  Membrane Capacitance, 20*g_L in nF
        double C_m;

        //!  Excitatory reversal Potential
        double E_ex;

        //!  Inhibitory reversal Potential
        double E_in;

        //!  Leak reversal Potential (aka resting potential)
        double E_L;

        double tau_syn_ex_rec1;

        //!  Synaptic Time Constant Excitatory Synapse
        double tau_syn_ex_rec0;

        //!  Synaptic Time Constant for Inhibitory Synapse
        double tau_syn_in;

        //! Rate estimation (filtering) time constant
        double tau_w;

        //!  Fractional change of synaptic activation
        double rho;

        //!  constant external input current
        double I_e;

        double __gsl_error_tol;
        /** Initialize parameters to their default values. */
        Parameters_();
    };

    /**
    * Dynamic state of the neuron.
    *
    *
    *
    * These are the state variables that are advanced in time by calls to
    * @c update(). In many models, some or all of them can be set by the user
    * through @c `node.set()`. The state variables are initialized from the model
    * prototype when the node is created. State variables are reset by @c ResetNetwork.
    *
    * @note State_ need neither copy constructor nor @c operator=(), since
    *       all its members are copied properly by the default copy constructor
    *       and assignment operator. Important:
    *       - If State_ contained @c Time members, you need to define the
    *         assignment operator to recalibrate all members of type @c Time . You
    *         may also want to define the assignment operator.
    *       - If State_ contained members that cannot copy themselves, such
    *         as C-style arrays, you need to define the copy constructor and
    *         assignment operator to copy those members.
    */
    struct State_{
        //! Symbolic indices to the elements of the state vector y
        enum StateVecElems{

            // numeric solver state variables
            V_m,
            g_ex__X__spikeExcRec1,
            g_ex__X__spikeExcRec0,
            g_in__X__spikeInh,
            // trace variables
            STATE_VEC_SIZE
        };
        //! state vector, must be C-array for GSL solver
        double ode_state[STATE_VEC_SIZE];

        // state variables from state block

        //!  counts number of tick during the refractory period
        long r;

        State_();
    };

    /**
    * Internal variables of the neuron.
    *
    *
    *
    * These variables must be initialized by @c calibrate, which is called before
    * the first call to @c update() upon each call to @c Simulate.
    * @node Variables_ needs neither constructor, copy constructor or assignment operator,
    *       since it is initialized by @c calibrate(). If Variables_ has members that
    *       cannot destroy themselves, Variables_ will need a destructor.
    */
    struct Variables_ {
        // recording variables, for debugging and visualization
        std::vector<double> recorded_times;
        std::vector<long> recorded_input_ids;
        std::vector<double> recorded_activation_traces;
        std::vector<double> recorded_rates;

        double preSynActivationTraces[10000];  // synaptic activation traces, for each incoming synapse
        double preSynWeights[10000]; // weights of the current/last spike from each synapse
        bool isExcRec1[10000];

        std::list<TraceTracker_> spikeEvents;  // TODO add comm
        std::set<unsigned long> activeSources;  // TODO add comm

        double t_end_last_trial;  // end timepoint of last trial. Used to ignore all incoming spikes from prev. trial.

        //!  refractory time in steps
        long RefractoryCounts;

        //! estimated rate (filtered)
        double rate;
        double __rate_kernel;

        double __h;

        double __P__g_ex__X__spikeRec1__g_Rec1__X__spikeRec1;

        double __P__g_ex__X__spikeRec0__g_Rec0__X__spikeRec0;

        double __P__g_in__X__spikeInh__g_in__X__spikeInh;

        // trace variables
    };

    /**
      * Buffers of the neuron.
      * Usually buffers for incoming spikes and data logged for analog recorders.
      * Buffers must be initialized by @c init_buffers_(), which is called before
      * @c calibrate() on the first call to @c Simulate after the start of NEST,
      * ResetKernel or ResetNetwork.
      * @node Buffers_ needs neither constructor, copy constructor or assignment operator,
      *       since it is initialized by @c init_nodes_(). If Buffers_ has members that
      *       cannot destroy themselves, Buffers_ will need a destructor.
      */
    struct Buffers_ {
        Buffers_(ShouvalNeuron &);
        Buffers_(const Buffers_ &, ShouvalNeuron &);

        /** Logger for all analog data */
        nest::UniversalDataLogger<ShouvalNeuron> logger_;

        inline nest::RingBuffer& get_spikeInh() {return spikeInh;}
        //!< Buffer incoming nSs through delay, as sum
        nest::RingBuffer spikeInh;
        double spikeInh_grid_sum_;

        inline nest::RingBuffer& get_spikeExc() {return spikeExc;}
        //!< Buffer incoming nSs through delay, as sum
        nest::RingBuffer spikeExc;
        double spikeExc_grid_sum_;

        //!< Buffer incoming pAs through delay, as sum
        nest::RingBuffer I_stim;
        inline nest::RingBuffer& get_I_stim() {return I_stim;}
        double I_stim_grid_sum_;
        /** GSL ODE stuff */
        gsl_odeiv_step* __s;    //!< stepping function
        gsl_odeiv_control* __c; //!< adaptive stepsize control function
        gsl_odeiv_evolve* __e;  //!< evolution function
        gsl_odeiv_system __sys; //!< struct describing system
//        gsl_odeiv2_step* __s;    //!< stepping function
//        gsl_odeiv2_control* __c; //!< adaptive stepsize control function
//        gsl_odeiv2_evolve* __e;  //!< evolution function
//        gsl_odeiv2_system __sys; //!< struct describing system

        // IntergrationStep_ should be reset with the neuron on ResetNetwork,
        // but remain unchanged during calibration. Since it is initialized with
        // step_, and the resolution cannot change after nodes have been created,
        // it is safe to place both here.
        double __step;             //!< step size in ms
        double __integration_step; //!< current integration time step, updated by GSL
    };

    /* getters/setters for state block */
    inline long get_r() const {
        return S_.r;
    }
    inline void set_r(const long __v) {
        S_.r = __v;
    }
/* getters/setters for initial values block (excluding functions) */
    inline double get_V_m() const {
        return S_.ode_state[State_::V_m];
    }
    inline void set_V_m(const double __v) {
        S_.ode_state[State_::V_m] = __v;
    }

    inline double get_g_ex__X__spikeExcRec0() const {
        return S_.ode_state[State_::g_ex__X__spikeExcRec0];
    }
    inline void set_g_ex__X__spikeExcRec0(const double __v) {
        S_.ode_state[State_::g_ex__X__spikeExcRec0] = __v;
    }

    inline double get_g_in__X__spikeInh() const {
        return S_.ode_state[State_::g_in__X__spikeInh];
    }
    inline void set_g_in__X__spikeInh(const double __v) {
        S_.ode_state[State_::g_in__X__spikeInh] = __v;
    }

    inline double get_g_ex__X__spikeExcRec1() const {
        return S_.ode_state[State_::g_ex__X__spikeExcRec1];
    }
    inline void set_g_ex__X__spikeExcRec1(const double __v) {
        S_.ode_state[State_::g_ex__X__spikeExcRec1] = __v;
    }

    /* getters/setters for parameters */
    inline void set_logOutput(const bool __v) {
        P_.logOutput = __v;
    }

    inline bool get_logOutput() const {
        return P_.logOutput;
    }

    inline double get_V_th() const {
        return P_.V_th;
    }
    inline void set_V_th(const double __v) {
        P_.V_th = __v;
    }

    inline double get_V_reset() const {
        return P_.V_reset;
    }
    inline void set_V_reset(const double __v) {
        P_.V_reset = __v;
    }

    inline double get_t_ref() const {
        return P_.t_ref;
    }
    inline void set_t_ref(const double __v) {
        P_.t_ref = __v;
    }

    inline double get_g_L() const {
        return P_.g_L;
    }
    inline void set_g_L(const double __v) {
        P_.g_L = __v;
    }

    inline double get_C_m() const {
        return P_.C_m;
    }
    inline void set_C_m(const double __v) {
        P_.C_m = __v;
    }

    inline double get_E_ex() const {
        return P_.E_ex;
    }
    inline void set_E_ex(const double __v) {
        P_.E_ex = __v;
    }

    inline double get_E_in() const {
        return P_.E_in;
    }
    inline void set_E_in(const double __v) {
        P_.E_in = __v;
    }

    inline double get_E_L() const {
        return P_.E_L;
    }
    inline void set_E_L(const double __v) {
        P_.E_L = __v;
    }

    inline double get_tau_syn_ex_rec1() const {
        return P_.tau_syn_ex_rec1;
    }
    inline void set_tau_syn_ex_rec1(const double __v) {
        P_.tau_syn_ex_rec1 = __v;
    }

    inline double get_tau_syn_ex() const {
        return P_.tau_syn_ex_rec0;
    }
    inline void set_tau_syn_ex(const double __v) {
        P_.tau_syn_ex_rec0 = __v;
    }

    inline double get_tau_syn_in() const {
        return P_.tau_syn_in;
    }
    inline void set_tau_syn_in(const double __v) {
        P_.tau_syn_in = __v;
    }

    inline double get_tau_w() const {
        return P_.tau_w;
    }
    inline void set_tau_w(const double __v) {
        P_.tau_w = __v;
    }

    inline double get_rate() const {
        return V_.rate;
    }
    inline void set_rate(const double __v) {
        V_.rate = __v;
    }

    inline double get_rho() const {
        return P_.rho;
    }
    inline void set_rho(const double __v) {
        P_.rho = __v;
    }

    inline double get_I_e() const {
        return P_.I_e;
    }
    inline void set_I_e(const double __v) {
        P_.I_e = __v;
    }
/* getters/setters for parameters */


    inline long get_RefractoryCounts() const {
        return V_.RefractoryCounts;
    }
    inline void set_RefractoryCounts(const long __v) {
        V_.RefractoryCounts = __v;
    }

    inline double get___h() const {
        return V_.__h;
    }
    inline void set___h(const double __v) {
        V_.__h = __v;
    }

    inline double get___P__g_ex__X__spikeExc__g_ex__X__spikeExc() const {
        return V_.__P__g_ex__X__spikeRec0__g_Rec0__X__spikeRec0;
    }
    inline void set___P__g_ex__X__spikeExc__g_ex__X__spikeExc(const double __v) {
        V_.__P__g_ex__X__spikeRec0__g_Rec0__X__spikeRec0 = __v;
    }

    inline double get___P__g_in__X__spikeInh__g_in__X__spikeInh() const {
        return V_.__P__g_in__X__spikeInh__g_in__X__spikeInh;
    }
    inline void set___P__g_in__X__spikeInh__g_in__X__spikeInh(const double __v) {
        V_.__P__g_in__X__spikeInh__g_in__X__spikeInh = __v;
    }
/* getters/setters for functions */
    inline double get_I_syn_exc() const {
        return S_.ode_state[State_::g_ex__X__spikeExcRec0] * (S_.ode_state[State_::V_m] - P_.E_ex);
    }

    inline double get_I_syn_inh() const {
        return S_.ode_state[State_::g_in__X__spikeInh] * (S_.ode_state[State_::V_m] - P_.E_in);
    }

    inline double get_g_ex() const {
        /**
         * This is the conductance g_ex...
         */
        return (S_.ode_state[State_::g_ex__X__spikeExcRec0] * (S_.ode_state[State_::V_m] - P_.E_ex))
               / (S_.ode_state[State_::V_m] - P_.E_ex);
    }

    inline double get_g_in() const {
        /**
         * This is the conductance g_ex...
         */
        return (S_.ode_state[State_::g_in__X__spikeInh] * (S_.ode_state[State_::V_m] - P_.E_in)) /
               (S_.ode_state[State_::V_m] - P_.E_in);
    }

    inline double get_I_leak() const {
        return P_.g_L * (S_.ode_state[State_::V_m] - P_.E_L);
    }

    inline double get_t_end_last_trial() const {
        return V_.t_end_last_trial;
    }

    inline void set_t_end_last_trial(const double __v) {
        V_.t_end_last_trial = __v;
    }

    inline std::vector< double > get_recorded_times() const {
        return V_.recorded_times;
    }

    inline void set_recorded_times(const std::vector< double > __v) {
        V_.recorded_times = __v;
    }

    inline std::vector< long > get_recorded_input_ids() const {
        return V_.recorded_input_ids;
    }

    inline void set_recorded_input_ids(const std::vector< long > __v) {
        V_.recorded_input_ids = __v;
    }

    inline std::vector< double > get_recorded_activation_traces() const {
        return V_.recorded_activation_traces;
    }

    inline void set_recorded_activation_traces(const std::vector< double > __v) {
        V_.recorded_activation_traces = __v;
    }

    inline std::vector< double > get_recorded_rates() const {
        return V_.recorded_rates;
    }

    inline void set_recorded_rates(const std::vector< double > __v) {
        V_.recorded_rates = __v;
    }

    /* getters/setters for input buffers */

    inline nest::RingBuffer& get_spikeInh() {return B_.get_spikeInh();};
    inline nest::RingBuffer& get_spikeExc() {return B_.get_spikeExc();};
    inline nest::RingBuffer& get_I_stim() {return B_.get_I_stim();};

    /**
    * @defgroup pif_members Member variables of neuron model.
    * Each model neuron should have precisely the following four data members,
    * which are one instance each of the parameters, state, buffers and variables
    * structures. Experience indicates that the state and variables member should
    * be next to each other to achieve good efficiency (caching).
    * @note Devices require one additional data member, an instance of the @c Device
    *       child class they belong to.
    * @{
    */
    Parameters_ P_;  //!< Free parameters.
    State_      S_;  //!< Dynamic state.
    Variables_  V_;  //!< Internal Variables
    Buffers_    B_;  //!< Buffers.

    //! Mapping of recordables names to access functions
    static nest::RecordablesMap<ShouvalNeuron> recordablesMap_;

    friend int ttl_neuron_dynamics( double, const double y[], double f[], void* pnode );


/** @} */
}; /* neuron ShouvalNeuron */

inline nest::port ShouvalNeuron::send_test_event(
        nest::Node& target, nest::rport receptor_type, nest::synindex, bool){
    // You should usually not change the code in this function.
    // It confirms that the target of connection @c c accepts @c nest::SpikeEvent on
    // the given @c receptor_type.
    nest::SpikeEvent e;
    e.set_sender(*this);
    return target.handles_test_event(e, receptor_type);
}

inline nest::port ShouvalNeuron::handles_test_event(nest::SpikeEvent&, nest::port receptor_type){

    // You should usually not change the code in this function.
    // It confirms to the connection management system that we are able
    // to handle @c SpikeEvent on port 0. You need to extend the function
    // if you want to differentiate between input ports.
//    if (receptor_type != 0)
//        throw nest::UnknownReceptorType(receptor_type, get_name());
//    return 0;
    if ( receptor_type < 0 || receptor_type > 1 )
    {
        throw nest::IncompatibleReceptorType( receptor_type, get_name(), "SpikeEvent" );
    }

//    P_.has_connections_ = true;
    return receptor_type;
}



inline nest::port ShouvalNeuron::handles_test_event(
        nest::CurrentEvent&, nest::port receptor_type){
    // You should usually not change the code in this function.
    // It confirms to the connection management system that we are able
    // to handle @c CurrentEvent on port 0. You need to extend the function
    // if you want to differentiate between input ports.
    if (receptor_type != 0)
        throw nest::UnknownReceptorType(receptor_type, get_name());
    return 0;
}

inline nest::port ShouvalNeuron::handles_test_event(
        nest::DataLoggingRequest& dlr, nest::port receptor_type){
    // You should usually not change the code in this function.
    // It confirms to the connection management system that we are able
    // to handle @c DataLoggingRequest on port 0.
    // The function also tells the built-in UniversalDataLogger that this node
    // is recorded from and that it thus needs to collect data during simulation.
    if (receptor_type != 0)
        throw nest::UnknownReceptorType(receptor_type, get_name());

    return B_.logger_.connect_logging_device(dlr, recordablesMap_);
}

// TODO call get_status on used or internal components
inline void ShouvalNeuron::get_status(DictionaryDatum &__d) const{

    // parameters
    def<double>(__d, "V_th", get_V_th());
    def<double>(__d, "V_reset", get_V_reset());
    def<double>(__d, "t_ref", get_t_ref());
    def<double>(__d, "g_L", get_g_L());
    def<double>(__d, "C_m", get_C_m());
    def<double>(__d, "E_ex", get_E_ex());
    def<double>(__d, "E_in", get_E_in());
    def<double>(__d, "E_L", get_E_L());
    def<double>(__d, "tau_syn_ex_rec1", get_tau_syn_ex_rec1());
    def<double>(__d, "tau_syn_ex_rec0", get_tau_syn_ex());
    def<double>(__d, "tau_syn_in", get_tau_syn_in());
    def<double>(__d, "tau_w", get_tau_w());
    def<double>(__d, "rho", get_rho());
    def<double>(__d, "I_e", get_I_e());
    def<double>(__d, "logOutput", get_logOutput());

    // initial values for state variables not in ODE or kernel
    def<long>(__d, "r", get_r());
    def<double>(__d, "rate", get_rate());

    // initial values for state variables in ODE or kernel
    def<double>(__d, "V_m", get_V_m());
    def<double>(__d, "g_ex__X__spikeExcRec0", get_g_ex__X__spikeExcRec0());
    def<double>(__d, "g_ex__X__spikeExcRec1", get_g_ex__X__spikeExcRec1());
    def<double>(__d, "g_in__X__spikeInh", get_g_in__X__spikeInh());

    def<double>(__d, "t_end_last_trial", get_t_end_last_trial());
    def<std::vector<double>>(__d, "recorded_times", get_recorded_times());
    def<std::vector<long>>(__d, "recorded_input_ids", get_recorded_input_ids());
    def<std::vector<double>>(__d, "recorded_activation_traces", get_recorded_activation_traces());
    def<std::vector<double>>(__d, "recorded_rates", get_recorded_rates());

    Shouval_Archiving_Node::get_status( __d );

    (*__d)[nest::names::recordables] = recordablesMap_.get_list();

    def< double >(__d, nest::names::gsl_error_tol, P_.__gsl_error_tol);
    if ( P_.__gsl_error_tol <= 0. ){
        throw nest::BadProperty( "The gsl_error_tol must be strictly positive." );
    }
}

inline void ShouvalNeuron::set_status(const DictionaryDatum &__d){
    // parameters
    bool tmp_logOutput = get_logOutput();
    updateValue<bool>(__d, "logOutput", tmp_logOutput);

    double tmp_V_th = get_V_th();
    updateValue<double>(__d, "V_th", tmp_V_th);


    double tmp_V_reset = get_V_reset();
    updateValue<double>(__d, "V_reset", tmp_V_reset);


    double tmp_t_ref = get_t_ref();
    updateValue<double>(__d, "t_ref", tmp_t_ref);


    double tmp_g_L = get_g_L();
    updateValue<double>(__d, "g_L", tmp_g_L);

    double tmp_C_m = get_C_m();
    updateValue<double>(__d, "C_m", tmp_C_m);


    double tmp_E_ex = get_E_ex();
    updateValue<double>(__d, "E_ex", tmp_E_ex);


    double tmp_E_in = get_E_in();
    updateValue<double>(__d, "E_in", tmp_E_in);


    double tmp_E_L = get_E_L();
    updateValue<double>(__d, "E_L", tmp_E_L);

    double tmp_tau_syn_ex_rec1 = get_tau_syn_ex_rec1();
    updateValue<double>(__d, "tau_syn_ex_rec1", tmp_tau_syn_ex_rec1);

    double tmp_tau_syn_ex = get_tau_syn_ex();
    updateValue<double>(__d, "tau_syn_ex_rec0", tmp_tau_syn_ex);


    double tmp_tau_syn_in = get_tau_syn_in();
    updateValue<double>(__d, "tau_syn_in", tmp_tau_syn_in);

    double tmp_tau_w = get_tau_w();
    updateValue<double>(__d, "tau_w", tmp_tau_w);

    double tmp_rho = get_rho();
    updateValue<double>(__d, "rho", tmp_rho);

    double tmp_I_e = get_I_e();
    updateValue<double>(__d, "I_e", tmp_I_e);

    // initial values for state variables not in ODE or kernel
    long tmp_r = get_r();
    updateValue<long>(__d, "r", tmp_r);

    double tmp_rate = get_rate();
    updateValue<double>(__d, "rate", tmp_rate);

    // initial values for state variables in ODE or kernel
    double tmp_V_m = get_V_m();
    updateValue<double>(__d, "V_m", tmp_V_m);

    double tmp_g_ex__X__spikeExc = get_g_ex__X__spikeExcRec0();
    updateValue<double>(__d, "g_ex__X__spikeExcRec0", tmp_g_ex__X__spikeExc);

    double tmp_g_ex__X__spikeExcRec1 = get_g_ex__X__spikeExcRec1();
    updateValue<double>(__d, "g_ex__X__spikeExcRec1", tmp_g_ex__X__spikeExcRec1);

    double tmp_g_in__X__spikeInh = get_g_in__X__spikeInh();
    updateValue<double>(__d, "g_in__X__spikeInh", tmp_g_in__X__spikeInh);

    double tmp__t_end_last_trial = get_t_end_last_trial();
    updateValue<double>(__d, "t_end_last_trial", tmp__t_end_last_trial);

    std::vector<double> tmp_recorded_times = get_recorded_times();
    std::vector<long> tmp_recorded_input_ids = get_recorded_input_ids();
    std::vector<double> tmp_recorded_activation_traces = get_recorded_activation_traces();
    std::vector<double> tmp_recorded_rates = get_recorded_rates();

    // We now know that (ptmp, stmp) are consistent. We do not
    // write them back to (P_, S_) before we are also sure that
    // the properties to be set in the parent class are internally
    // consistent.
    Shouval_Archiving_Node::set_status(__d);

    // if we get here, temporaries contain consistent set of properties
    set_logOutput(tmp_logOutput);
    set_V_th(tmp_V_th);
    set_V_reset(tmp_V_reset);
    set_t_ref(tmp_t_ref);
    set_g_L(tmp_g_L);
    set_C_m(tmp_C_m);
    set_E_ex(tmp_E_ex);
    set_E_in(tmp_E_in);
    set_E_L(tmp_E_L);
    set_tau_syn_ex_rec1(tmp_tau_syn_ex_rec1);
    set_tau_syn_ex(tmp_tau_syn_ex);
    set_tau_syn_in(tmp_tau_syn_in);
    set_tau_w(tmp_tau_w);
    set_rho(tmp_rho);
    set_I_e(tmp_I_e);
    set_r(tmp_r);
    set_rate(tmp_rate);
    set_V_m(tmp_V_m);

    set_g_ex__X__spikeExcRec0(tmp_g_ex__X__spikeExc);
    set_g_in__X__spikeInh(tmp_g_in__X__spikeInh);
    set_g_ex__X__spikeExcRec1(tmp_g_ex__X__spikeExcRec1);

    //! reset all recording vectors upon change in P_.logOutput
    if (P_.logOutput)
    {
        tmp_recorded_times.clear();
        tmp_recorded_input_ids.clear();
        tmp_recorded_activation_traces.clear();
        tmp_recorded_rates.clear();
    }

    set_t_end_last_trial(tmp__t_end_last_trial);
    set_recorded_times(tmp_recorded_times);
    set_recorded_input_ids(tmp_recorded_input_ids);
    set_recorded_activation_traces(tmp_recorded_activation_traces);
    set_recorded_rates(tmp_recorded_rates);

    updateValue< double >(__d, nest::names::gsl_error_tol, P_.__gsl_error_tol);
    if ( P_.__gsl_error_tol <= 0. ){
        throw nest::BadProperty( "The gsl_error_tol must be strictly positive." );
    }

};

#endif /* #ifndef TTL_NEURON */
#endif /* HAVE GSL */