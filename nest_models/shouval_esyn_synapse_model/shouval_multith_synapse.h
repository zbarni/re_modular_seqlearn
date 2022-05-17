
/*
*  shouval_multith_synapse.h
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
*  2020-10-29 13:15:31.388998
*/

#ifndef SHOUVAL_CONNECTION_H
#define SHOUVAL_CONNECTION_H

// C++ includes:
#include <cmath>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>
//#include <cstdio>
#include <sstream>


// Includes from nestkernel:
#include "common_synapse_properties.h"
#include "connection.h"
#include "connector_model.h"
#include "event.h"
//#include "logging.h"

// Includes from sli:
#include "dictdatum.h"
#include "dictutils.h"


/** @BeginDocumentation

Authors
+++++++

Zajzon


**/

#define DYAD_POST_TYPE

namespace nest {

/** @BeginDocumentation

**/
    struct SolverParams_
    {
        SolverParams_() {};
        SolverParams_(const SolverParams_ &sp)
        {
            params[theta] = sp.params[theta];
            params[norm_traces] = sp.params[norm_traces];
            params[tau_w] = sp.params[tau_w];
            params[Tp_max_] = sp.params[Tp_max_];
            params[Td_max_] = sp.params[Td_max_];
            params[tau_ltp_] = sp.params[tau_ltp_];
            params[tau_ltd_] = sp.params[tau_ltd_];
            params[eta_ltp_] = sp.params[eta_ltp_];
            params[eta_ltd_] = sp.params[eta_ltd_];
            params[reward_on] = sp.params[reward_on];
            params[refractory_on] = sp.params[refractory_on];
        };

        enum SolverParamsVecElems
        {
            theta,
            norm_traces,
            tau_w,
            Tp_max_,
            Td_max_,
            tau_ltp_,
            tau_ltd_,
            eta_ltp_,
            eta_ltd_,
            learn_rate,
            reward_on,
            refractory_on,

            STATE_VEC_SIZE
        };
        double params[STATE_VEC_SIZE];
    };

    struct State_
    {
        enum StateVecElems
        {
            // numeric solver state variables
            w,
            trace_ltp,
            trace_ltd,
            rate_pre,
            rate_post,

            STATE_VEC_SIZE
        };

        //! state vector, must be C-array for GSL solver
        double y_[STATE_VEC_SIZE];

        State_() {};
        State_( const State_& s) {
            y_[w] = s.y_[w];
            y_[trace_ltp] = s.y_[trace_ltp];
            y_[trace_ltd] = s.y_[trace_ltd];
            y_[rate_pre] = s.y_[rate_pre];
            y_[rate_post] = s.y_[rate_post];
        };
    };

    extern "C" inline int ShouvalConnectionMultiThDynamics(double t, const double y[], double f[], void* psyn)
    {
        typedef nest::State_ S;
        const SolverParams_& sP = *( reinterpret_cast< SolverParams_* > ( psyn ) );

        const double& rate_pre = y[ S::rate_pre ];
        const double& rate_post = y[ S::rate_post ];

        double eff_saturation_ltp;
        double eff_saturation_ltd;
        double eff_tc_ltp;
        double eff_tc_ltd;

        //! WITH NORMALIZATION
        if (sP.params[SolverParams_::norm_traces] > 0.5) {
            assert(false);  // temporary
            double hebbian_ltp = sP.params[SolverParams_::eta_ltp_] * rate_pre * rate_post;
            double hebbian_ltd = sP.params[SolverParams_::eta_ltd_] * rate_pre * rate_post;

            //! there's a weird minimum threshold for the H term, 10 Hz for recurrent and 20 Hz for FF
            double rate_th = sP.params[SolverParams_::theta] * sP.params[SolverParams_::tau_w];
            if (rate_pre <= rate_th or rate_post <= rate_th) {
                hebbian_ltp = 0.;
                hebbian_ltd = 0.;
            }

            eff_saturation_ltp = sP.params[SolverParams_::Tp_max_] * hebbian_ltp /
                                        (sP.params[SolverParams_::Tp_max_] + hebbian_ltp);
            eff_tc_ltp = sP.params[SolverParams_::Tp_max_] * sP.params[SolverParams_::tau_ltp_] /
                                (sP.params[SolverParams_::Tp_max_] + hebbian_ltp);

            eff_saturation_ltd = sP.params[SolverParams_::Td_max_] * hebbian_ltd /
                                        (sP.params[SolverParams_::Td_max_] + hebbian_ltd);
            eff_tc_ltd = sP.params[SolverParams_::Td_max_] * sP.params[SolverParams_::tau_ltd_] /
                                (sP.params[SolverParams_::Td_max_] + hebbian_ltd);

            // update traces
            f[ S::trace_ltp ] = -1 / eff_tc_ltp * (y[ S::trace_ltp ] - eff_saturation_ltp);
            f[ S::trace_ltd ] = -1 / eff_tc_ltd * (y[ S::trace_ltd ] - eff_saturation_ltd);
        }
        else
        {
            //! WITHOUT NORMALIZATION - as in Cone & Shouval Matlab
            double hebbian_ltp = sP.params[SolverParams_::eta_ltp_] * rate_pre * rate_post;
            double hebbian_ltd = sP.params[SolverParams_::eta_ltd_] * rate_pre * rate_post;

            //! there's a weird minimum threshold for the H term, 10 Hz for recurrent and 20 Hz for FF
            double rate_th = sP.params[SolverParams_::theta];
            if (rate_pre <= rate_th or rate_post <= rate_th) {
                hebbian_ltp = 0.;
                hebbian_ltd = 0.;
            }

            f[ S::trace_ltp ] = (- y[ S::trace_ltp ]
                                 + hebbian_ltp * (sP.params[SolverParams_::Tp_max_] - y[ S::trace_ltp ]) )
                                / sP.params[SolverParams_::tau_ltp_];
            f[ S::trace_ltd ] = (- y[ S::trace_ltd ]
                                 + hebbian_ltd * (sP.params[SolverParams_::Td_max_] - y[ S::trace_ltd ]) )
                                / sP.params[SolverParams_::tau_ltd_];
        }

        f[ S::rate_pre ] = -rate_pre / sP.params[ SolverParams_::tau_w ];
        f[ S::rate_post ] = -rate_post / sP.params[ SolverParams_::tau_w ];

        if (sP.params[SolverParams_::refractory_on] > 1e-5) {
            // refractory on!
            f[ S::trace_ltp ] = 0;
            f[ S::trace_ltd ] = 0;
        }

        // reward on!
        if (sP.params[SolverParams_::reward_on] > 1e-5) {
            f[ S::w ] = sP.params[SolverParams_::learn_rate] * (y[ S::trace_ltp ] - y[ S::trace_ltd ]);
        }
        else {
            f[ S::w ] = 0;
        }
        return GSL_SUCCESS;
    }

/** @BeginDocumentation

**/

/**
 * Class representing a static connection.
 */
    template< typename targetidentifierT >
    class ShouvalConnectionMultiTh : public Connection< targetidentifierT >
    {

    public:
//  typedef ShouvalConnectionMultiThCommonSynapseProperties CommonPropertiesType;
        typedef CommonSynapseProperties CommonPropertiesType;
        typedef Connection<targetidentifierT> ConnectionBase;

        /**
        * Default constructor.
        *
        * Sets default values for all parameters (skipping common properties).
        *
        * Needed by GenericConnectorModel.
        */
        ShouvalConnectionMultiTh();


        /**
        * Copy constructor from a property object.
        *
        * Sets default values for all parameters (skipping common properties).
        *
        * Needs to be defined properly in order for GenericConnector to work.
        */
        ShouvalConnectionMultiTh(const ShouvalConnectionMultiTh &);

//  void init_buffers_();
        void init_internals_block_symbols();
        void evolve_dynamics_X(const double, const int);
        void fill_missing_samples(const double);
        void log_complete_state(std::string);
        void log_message(const thread tid, std::ostringstream &msg);
//        void reset_internal_state( std::ostringstream &msg );
        void reset_internal_state( );

        // Explicitly declare all methods inherited from the dependent base
        // ConnectionBase. This avoids explicit name prefixes in all places these
        // functions are used. Since ConnectionBase depends on the template parameter,
        // they are not automatically found in the base class.
        using ConnectionBase::get_delay_steps;
        using ConnectionBase::set_delay_steps;
        using ConnectionBase::get_delay;
        using ConnectionBase::set_delay;
        using ConnectionBase::get_rport;
        using ConnectionBase::get_target;

        /**
         * Get all properties of this connection and put them into a dictionary.
         */
        void get_status( DictionaryDatum &d ) const;

        /**
         * Set properties of this connection from the values given in dictionary.
         */
        void set_status( const DictionaryDatum &d, ConnectorModel &cm );

        /**
       * Send an event to the receiver of this connection.
       * \param e The event to send
       * \param cp common properties of all synapses (empty).
       */
        void send( Event &e, thread tid, const CommonSynapseProperties &cp );

        class ConnTestDummyNode : public ConnTestDummyNodeBase
        {
        public:
            // Ensure proper overriding of overloaded virtual functions.
            // Return values from functions are ignored.
            using ConnTestDummyNodeBase::handles_test_event;

            port
            handles_test_event( SpikeEvent &, rport ) {
                return invalid_port_;
            }
        };

        void
        check_connection( Node &s, Node &t, rport receptor_type, const CommonPropertiesType & )
        {
            ConnTestDummyNode dummy_target;
            ConnectionBase::check_connection_(dummy_target, s, t, receptor_type);
            t.register_stdp_connection(t_lastspike_ - get_delay(), get_delay());
        }

        inline void set_weight(double w) {
            weight_ = w;
        }

        void
        print_state(std::ostringstream &msg) {
            msg << "\t\tState:"
                << "\n\t\t\tT_ltp / T_ltd: " << S_.y_[State_::trace_ltp] << " / " << S_.y_[State_::trace_ltd]
                << "\n\t\t\tw: " << S_.y_[State_::w] << "\n";
        }

        /**
         * Sample and log/store LTP and LTD traces, as well as the weight, at regular intervals to stdout.

         * @param cur_tp
         * @param ref_tp
         */
        void
        store_interval_sampling_state(const double tp)
        {
//          TODO here we could ensure that tp%sampling_interval == 0, so only valid times are recorded
//          TODO we could also preset the array sizes as push_back is rather inefficient

            if (P_.sampling_interval >= 1.0)
            {
                double q = tp / P_.sampling_interval;
                if (q == rintf(q))
                {
                    V_.sampled_times.push_back(tp);
                    V_.sampled_ltp.push_back(S_.y_[State_::trace_ltp]);
                    V_.sampled_ltd.push_back(S_.y_[State_::trace_ltd]);
                    V_.sampled_weights.push_back(weight_);
                }
            }
        }

        /**
         *
         * @param t_spike
         * @param msg
         * @return
         */
        bool check_and_advance_reward_counter(const double t_spike, std::ostringstream &msg)
        {
            // advance reward counter if current spike is beyond the next reward onset
            if (V_.__t_reward_idx + 1 < P_.reward_times.size() and
                t_spike >= P_.reward_times[V_.__t_reward_idx + 1])
            {
                ++V_.__t_reward_idx;
                msg << "advance reward counter: " << P_.reward_times[V_.__t_reward_idx] << "\n";
                return true;
            }
            return false;
        }

        void state_consistency_check_solver(const thread &tid, std::ostringstream &msg)
        {
            assert(B_vec_[tid]->__c);
            assert(B_vec_[tid]->__e);
            assert(B_vec_[tid]->__s);
            assert(&(B_vec_[tid]->__sys));
            assert(B_vec_[tid]->__sys.params);
            assert(B_vec_[tid]->__sys.params == &(this->SP_));

            const SolverParams_& sP = *( reinterpret_cast< SolverParams_* > ( B_vec_[tid]->__sys.params ) );

            msg << "\n\tSolver Consistency Check"
                << "\n\t\tB_vec_ @" << &B_vec_ << ";\tB_[tid] @" << B_vec_[tid]
                << ";\tB_->__sys.params: " << B_vec_[tid]->__sys.params
                << "\n\t\tSP: "
                << sP.params[SolverParams_::tau_w] << " "
                << sP.params[SolverParams_::eta_ltp_] << " "
                << sP.params[SolverParams_::eta_ltd_] << " "
                << sP.params[SolverParams_::Tp_max_] << " "
                << sP.params[SolverParams_::Td_max_] << " "
                << sP.params[SolverParams_::tau_ltp_] << " "
                << sP.params[SolverParams_::tau_ltd_] << " "
                << sP.params[SolverParams_::reward_on] << " "
                << sP.params[SolverParams_::refractory_on] << " "
                << "\n\t\tS: "
                << S_.y_[State_::rate_pre] << " "
                << S_.y_[State_::rate_post] << " "
                << S_.y_[State_::trace_ltp] << " "
                << S_.y_[State_::trace_ltd] << " "
                << S_.y_[State_::w] << " "

                << std::endl;
        }

        /**
         *
         */
        void check_trace_validity()
        {
//            double etol = 1e-3;
            double etol = 100.;
            if (S_.y_[State_::trace_ltp] < 0.)
            {
                if (std::abs(S_.y_[State_::trace_ltp]) < etol)
                {
                    S_.y_[State_::trace_ltp] = 0.;
                }
            }

            if (S_.y_[State_::trace_ltd] < 0.)
            {
                if (std::abs(S_.y_[State_::trace_ltd]) < etol)
                {
                    S_.y_[State_::trace_ltd] = 0.;
                }
            }
//            assert(S_.y_[State_::trace_ltd] >= 0. && "Trace can't be negative!");
//            assert(S_.y_[State_::trace_ltp] >= 0. && "Trace can't be negative!");
        }

        /**
         * Print error message if the sign of the weight has changed during update.
         * @param tmp_init_weight
         */
        void check_negative_weights(double tmp_init_weight, double ltp, double ltd,
                                    double t_start, double t_stop)
        {
            if (tmp_init_weight != S_.y_[State_::w] and S_.y_[State_::w] < 0.) {
//                std::cout << "ERRRRRRORRR? - negative weights\n\n"
//                             "check_negative_weights(): weight Init Vs New: "
//                          << tmp_init_weight << " : " << S_.y_[State_::w] << " in window ["
//                          << t_start << ", " << t_stop << "]\n" << std::flush;
//                std::cout << "check_negative_weights(): WEIGHTS CHANGED! reward on? "
//                          << SP_.params[SolverParams_::reward_on]
//                          << " ltp/d values: " << ltp << ", " << ltd
//                          << " vs new " << S_.y_[State_::trace_ltp] << ", " << S_.y_[State_::trace_ltd]
//                          <<  "\n" << std::flush;

                // TODO temporary!
                // set negative weights to 0. This is a temporary solution
                S_.y_[State_::w] = 0.;
            }
        }


    private:
        unsigned n_max_threads = 128;
        static const int n_max_samples = 1000;

        //! this is the (main) weight used when sending spikes. When `weight_update_once` == True, this variable does
        //! not change during the trial. S_[w] is used to compute the weight changes continuously, and at the end of
        //! the trial when the fake spike is set this variable is also updated: weight_ = S_[w].
        double weight_;
        double t_lastspike_;

        inline double get_w() const {
            return S_.y_[State_::w];
        }

        inline void set_w(const double __v) {
            S_.y_[State_::w] = __v;
            weight_ = __v;  // need to update the main weight_ var here for consistency!!!
        }

        inline double get_trace_ltp() const {
            return S_.y_[State_::trace_ltp];
        }

        inline void set_trace_ltp(const double __v) {
            S_.y_[State_::trace_ltp] = __v;
        }

        inline double get_trace_ltd() const {
            return S_.y_[State_::trace_ltd];
        }

        inline void set_trace_ltd(const double __v) {
            S_.y_[State_::trace_ltd] = __v;
        }

        inline double get_rate_pre() const {
            return S_.y_[State_::rate_pre];
        }

        inline void set_rate_pre(const double __v) {
            S_.y_[State_::rate_pre] = __v;
        }

        inline double get_rate_post() const {
            return S_.y_[State_::rate_post];
        }

        inline void set_rate_post(const double __v) {
            S_.y_[State_::rate_post] = __v;
        }

        /* getters/setters for parameters */
        inline double get_tau_w() const {
            return P_.tau_w;
        }

        inline void set_tau_w(const double __v) {
            P_.tau_w = __v;
        }

        inline double get_theta() const {
            return P_.theta;
        }

        inline void set_theta(const double __v) {
            P_.theta = __v;
        }

        inline double get_tau_ltp() const {
            return P_.tau_ltp;
        }

        inline void set_tau_ltp(const double __v) {
            P_.tau_ltp = __v;
        }

        inline double get_tau_ltd() const {
            return P_.tau_ltd;
        }

        inline void set_tau_ltd(const double __v) {
            P_.tau_ltd = __v;
        }

        inline double get_Tp_max() const {
            return P_.Tp_max;
        }

        inline void set_Tp_max(const double __v) {
            P_.Tp_max = __v;
        }

        inline double get_Td_max() const {
            return P_.Td_max;
        }

        inline void set_Td_max(const double __v) {
            P_.Td_max = __v;
        }

        inline double get_eta_ltp() const {
            return P_.eta_ltp;
        }

        inline void set_eta_ltp(const double __v) {
            P_.eta_ltp = __v;
        }

        inline double get_eta_ltd() const {
            return P_.eta_ltd;
        }

        inline void set_eta_ltd(const double __v) {
            P_.eta_ltd = __v;
        }

        inline double get_T_tr() const {
            return P_.T_tr;
        }

        inline void set_T_tr(const double __v) {
            P_.T_tr = __v;
        }

        inline double get_T_reward() const {
            return P_.T_reward;
        }

        inline void set_T_reward(const double __v) {
            P_.T_reward = __v;
        }

        inline double get_the_delay() const {
            return P_.the_delay;
        }

        inline void set_the_delay(const double __v) {
            P_.the_delay = __v;
        }

        inline double get_learn_rate() const {
            return P_.learn_rate;
        }

        inline double get_weight_update_once() const {
            return P_.weight_update_once;
        }

        inline double get_norm_traces() const {
            return P_.norm_traces;
        }

        inline double get_weights_ns() const {
            return P_.weights_ns;
        }

        inline void set_learn_rate(const double __v) {
            P_.learn_rate = __v;
        }

        inline std::vector< double > get_reward_times() const {
            return P_.reward_times;
        }

        inline void set_reward_times(const std::vector< double > __v) {
            P_.reward_times = __v;
        }

        //////
        inline double get_sampling_interval() const {
            return P_.sampling_interval;
        }

        inline void set_sampling_interval(const double __v) {
            P_.sampling_interval = __v;
        }

        //////
        inline std::vector< double > get_sampled_times() const {
            return V_.sampled_times;
        }

        inline void set_sampled_times(const std::vector< double > __v) {
            V_.sampled_times = __v;
        }

        //////
        inline std::vector< double > get_sampled_ltp() const {
            return V_.sampled_ltp;
        }

        inline void set_sampled_ltp(const std::vector< double > __v) {
            V_.sampled_ltp = __v;
        }

        //////
        inline std::vector< double > get_sampled_ltd() const {
            return V_.sampled_ltd;
        }

        inline void set_sampled_ltd(const std::vector< double > __v) {
            V_.sampled_ltd = __v;
        }

        //////
        inline std::vector< double > get_sampled_weights() const {
            return V_.sampled_weights;
        }

        inline void set_sampled_weights(const std::vector< double > __v) {
            V_.sampled_weights = __v;
        }

        //////
        inline std::vector< double > get_sampled_hebbian() const {
            return V_.sampled_hebbian;
        }

        inline void set_sampled_hebbian(const std::vector< double > __v) {
            V_.sampled_hebbian = __v;
        }

        struct Parameters_
        {

            //!  filtering time constant for firing rate estimation
            double tau_w;

            double theta;  //! firing rate threshold for plasticity

            double tau_ltp;

            double tau_ltd;

            double Tp_max;

            double Td_max;

            //!  Activation rate LTP trace
            double eta_ltp;

            //!  Activation rate LTD trace
            double eta_ltd;

            //!  Duration of refractory period for traces following neuromodulator presentation
            double T_tr;

            //!  duration of reward window
            double T_reward;

            //!  !! this is not mentioned in the paper!
            double the_delay;

            double learn_rate;

            double __gsl_error_tol;

            std::vector< double > reward_times;

            double sampling_interval;  // interval at which to sample traces / weights to log file

            double weight_update_once;  // buffer weights during trial and only update at the end, during fake_spike
            double norm_traces;  // normalize trace equations a
            double weights_ns;  // whether weight units are nS (otherwise microsS)

            /** Initialize parameters to their default values. */
            Parameters_() {};
        };

        struct ODEStruct1_ {
            ODEStruct1_();
            ODEStruct1_(const ODEStruct1_ &);

            /** GSL ODE stuff */
            gsl_odeiv_step* __s;    //!< stepping function
            gsl_odeiv_control* __c; //!< adaptive stepsize control function
            gsl_odeiv_evolve* __e;  //!< evolution function
//    std::vector < gsl_odeiv_evolve* > __e = std::vector< gsl_odeiv_evolve* >(256);  //!< evolution function
            gsl_odeiv_system __sys; //!< struct describing system

            // IntergrationStep_ should be reset with the neuron on ResetNetwork,
            // but remain unchanged during calibration. Since it is initialized with
            // step_, and the resolution cannot change after nodes have been created,
            // it is safe to place both here.
            double __step;             //!< step size in ms
            double __integration_step; //!< current integration time step, updated by GSL
        };


        struct Variables_ {
            double __h;

            unsigned __t_reward_idx;

            double __P__rate_pre__rate_pre;

            double __P__rate_post__rate_post;

            double t_last_reward;  // tp of last reward (from previous batch)
            std::vector< double > sampled_times;
            std::vector< double > sampled_ltp;
            std::vector< double > sampled_ltd;
            std::vector< double > sampled_weights;
            std::vector< double > sampled_hebbian;

            Node *unique_target;  // unique Node target of this synapse

            Variables_() {}
            Variables_(const Variables_ &v)
            {
                __h = v.__h;
                __t_reward_idx = v.__t_reward_idx;
                __P__rate_pre__rate_pre = v.__P__rate_pre__rate_pre;
                __P__rate_post__rate_post = v.__P__rate_post__rate_post;
                t_last_reward = v.t_last_reward;
                sampled_times = v.sampled_times;
                sampled_ltp = v.sampled_ltp;
                sampled_ltd = v.sampled_ltd;
                sampled_weights = v.sampled_weights;
                sampled_hebbian = v.sampled_hebbian;
                unique_target = v.unique_target;

            }

            Variables_(const Variables_ &original, const Variables_ &new_)
            {
                __h = original.__h;
                __t_reward_idx = original.__t_reward_idx;
                __P__rate_pre__rate_pre = original.__P__rate_pre__rate_pre;
                __P__rate_post__rate_post = original.__P__rate_post__rate_post;
                unique_target = original.unique_target;

                t_last_reward = new_.t_last_reward;
                sampled_times = new_.sampled_times;
                sampled_ltp = new_.sampled_ltp;
                sampled_ltd = new_.sampled_ltd;
                sampled_weights = new_.sampled_weights;
                sampled_hebbian = new_.sampled_hebbian;
            };
        };

        void init_buffers_(ODEStruct1_ *);
        void reset_buffers_(ODEStruct1_ *);
//        void init_buffers_();
        void _evolve_interval_block(const thread &,
                                    double &,
                                    const double &,
                                    const double &,
                                    ODEStruct1_ *,
                                    SolverParams_ *,
                                    bool,
                                    bool,
                                    std::ostringstream &);

        /**
        * Internal variables of the synapse.
        *
        *
        *
        * These variables must be initialized by @c calibrate, which is called before
        * the first call to @c update() upon each call to @c Simulate.
        * @node Variables_ needs neither constructor, copy constructor or assignment operator,
        *       since it is initialized by @c calibrate(). If Variables_ has members that
        *       cannot destroy themselves, Variables_ will need a destructor.
         */
        Parameters_ P_; //!< Free parameters.
        State_ S_;  //!< Dynamic state.
        Variables_ V_;  //!< Internal Variables
//        ODEStruct1_ *B_;  //!< Buffers / ODE integration variable
//        std::vector< ODEStruct1_ > B_vec_ = std::vector< ODEStruct1_ >(1);  //!< Buffers / ODE integration variable
        //!< Buffers / ODE integration variable
        std::vector< ODEStruct1_* > B_vec_ = std::vector< ODEStruct1_* >(n_max_threads);
        SolverParams_ SP_;
    };

////////////////////////////////////////////////
    template<typename targetidentifierT>
    inline void
    ShouvalConnectionMultiTh<targetidentifierT>::log_message(const thread tid, std::ostringstream &msg)
    {
        std::stringstream ss;
        std::stringstream header;
//        ss << "\n####################    thread  #" << tid << "  ######################\n";
        ss << msg.str();
//        ss << "\n///////////////////    thread  #" << tid << "  //////////////////////\n";

        header << "ShouvalConnectionMultiTh::log_msg( " << this << " )";
        // only log message for recorded connections
        if ( P_.sampling_interval >= 1. )
        {
            LOG(M_DEBUG, header.str(), ss.str());
        }
        msg.str(""); msg.clear();
    }

    template < typename targetidentifierT >
    void
    ShouvalConnectionMultiTh< targetidentifierT >::_evolve_interval_block(
            const thread &tid,
            double &tp_runner,
            const double &t_block_limit_right,  // used to be t_reward_on for first block
            const double &t_curspike_,
            ODEStruct1_ *B_,
            SolverParams_ *tmpSP,
            bool reward_window,
            bool refractory_window,
            std::ostringstream &msg)
    {
        bool sampling_enabled = (bool)(P_.sampling_interval >= 1.);
        SolverParams_ *__tmpSP_check = reinterpret_cast<SolverParams_*>(B_->__sys.params);

        double tp_solver_step;  // nr of timesteps for each solver iteration (init & update in each block)
        double tp_next_stop;  // timepoint of next stop (stepping), initialized & updated within each interval block

        int next_sample_idx;  // index (!, number) of the next sampling timepoint, relative to 0 (% P_.sampling_interval)
        double next_sample_tp;  // actual next sampling timepoint => P_.sampling_interval * next_sample_idx

        tp_next_stop = std::min(t_curspike_, t_block_limit_right);

        reset_buffers_(B_);
        state_consistency_check_solver(tid, msg);

        double start_weight = S_.y_[State_::w]; // starting weight, will check if changed only when allowed!?
        double start_ltp = S_.y_[State_::trace_ltp]; // starting weight, will check if changed only when allowed!?
        double start_ltd = S_.y_[State_::trace_ltd]; // starting weight, will check if changed only when allowed!?

        if (reward_window)
        {
            msg << "\n\t\tReward window ON! Setting reward flag for tmpSP @" << tmpSP;
            tmpSP->params[SolverParams_::reward_on] = 1;
            msg << ".... and set!" << std::endl;
//            log_message(tid, msg);
        }

        if (refractory_window)
        {
            msg << "\n\t\tRefractory window ON!";
            tmpSP->params[SolverParams_::refractory_on] = 1;
            // need to reset traces during refractory window !!!
            S_.y_[State_::trace_ltp] = 0.;
            S_.y_[State_::trace_ltd] = 0.;
        }

//        store_interval_sampling_state(tp_runner, tp_runner);  // store sample at beginning of block

        // from previous tp_runner, sample at regular intervals
        msg << "\n\t==> Starting while loop to move runner from " << tp_runner << " -> " << tp_next_stop;
        while (tp_runner < tp_next_stop)
        {
            if (sampling_enabled)
            {
                // calculate next sampling timepoint
                next_sample_idx = (int)(floor((tp_runner) / P_.sampling_interval) + 1);
                next_sample_tp = P_.sampling_interval * next_sample_idx;
            }
            else
            {
                // if sampling is not needed, we can simply compute all the way to the current spike / or right limit,
                // no need to stop at regular intervals inbetween
                next_sample_tp = tp_next_stop;
            }

            // TODO HERE @zbarni infinite loop?
            // calculate next stopping point, whether sampling tp or the final stop point for this block
            tp_solver_step = std::min(next_sample_tp, tp_next_stop) - tp_runner;

            msg << "\n\t\t[while] runner from [tp_runner] " << tp_runner << " -> " << tp_next_stop << " [tp_next_stop]"
                << " with tp_solver_step: " << tp_solver_step << " and tp_next_stop: " << tp_next_stop << "; sampling_enabled: " << sampling_enabled
                << "\n\t\t reward check (" << __tmpSP_check->params[SolverParams_::reward_on] << ") vs reward used (" << tmpSP->params[SolverParams_::reward_on] << ")"
                << "\n\t\t __tmpSP_check check @" << __tmpSP_check << " vs tmpSP used @" << tmpSP
                << "\n\t\t refractory check (" << __tmpSP_check->params[SolverParams_::refractory_on] << ") vs refractory used (" << tmpSP->params[SolverParams_::refractory_on] << ")"
                << std::endl;
            log_message(tid, msg);

            double t = 0.0;
            while (t < tp_solver_step) {
                const int status = gsl_odeiv_evolve_apply(B_->__e, B_->__c, B_->__s, &(B_->__sys), &t,
                                                          tp_solver_step, &(B_->__integration_step), S_.y_);
                check_trace_validity();
//                state_consistency_check_solver(tid, msg);

                if (status != GSL_SUCCESS) {
                    throw nest::GSLSolverFailure("ShouvalConnectionMultiTh", status);
                }
            }

            msg << "\n\t\t[after solver] ----------------------------------------"
                << "\n\t\t reward check (" << __tmpSP_check->params[SolverParams_::reward_on] << ") vs reward used (" << tmpSP->params[SolverParams_::reward_on] << ")"
                << "\n\t\t refractory check (" << __tmpSP_check->params[SolverParams_::refractory_on] << ") vs refractory used (" << tmpSP->params[SolverParams_::refractory_on] << ")" << std::endl;

            tp_runner += tp_solver_step;  // update runner with the evolved time from solver

            print_state(msg);
            log_message(tid, msg);

            check_negative_weights(start_weight, start_ltp, start_ltd, tp_runner - tp_solver_step, tp_runner);
            store_interval_sampling_state(tp_runner);  // store sample if possible
        }

        msg << "\n\t\tEnded while loop with runner at " << tp_runner << " ; tp_next_stop was " << tp_next_stop
            << "\n============================================================================================\n";

//        // set tp_runner to the right limit of the block - starting point of the next block
//        tp_runner = std::min(t_curspike_, t_block_limit_right);

        if (reward_window)
        {
            msg << "\n\t\tDeactivating reward window OFF! Resetting reward flag for tmpSP @" << tmpSP << std::endl;
            tmpSP->params[SolverParams_::reward_on] = 0;
//            log_message(tid, msg);
        }
        else
        {
            if (abs(start_weight - S_.y_[State_::w]) > 1e-5)
            {
                std::cout << msg.str();
                std::cout << "\nweight change: " << abs(start_weight - S_.y_[State_::w])
                          << "\ntime : " << tp_runner;
                msg << "\nweight change: " << abs(start_weight - S_.y_[State_::w])
                          << "\ntime : " << tp_runner;
            }
        }

        if (refractory_window)
        {
            msg << "\n\t\tDeactivating refractory  window OFF!";
            tmpSP->params[SolverParams_::refractory_on] = 0;
            // need to reset traces during refractory window !!!
            S_.y_[State_::trace_ltp] = 0;
            S_.y_[State_::trace_ltd] = 0;
        }
    }

    /**
     * Evolve synaptic dynamics while trying to log values at regular intervals, as much as possible.
     * @tparam targetidentifierT
     * @param t_spike
     * @param tid
     */
    template < typename targetidentifierT >
    void
    ShouvalConnectionMultiTh< targetidentifierT >::evolve_dynamics_X(const double t_spike, const int tid)
    {
        ODEStruct1_ *B_ = B_vec_[tid];

        {
            reset_buffers_(B_);
        }

        std::ostringstream msg;
        B_->__integration_step = nest::Time::get_resolution().get_ms();  // start with smallest step possible
        SolverParams_ *tmpSP = reinterpret_cast<SolverParams_*>(B_->__sys.params);
        // update some parameters so that ODE solver matches current values
        tmpSP->params[SolverParams_::theta] = P_.theta;
        tmpSP->params[SolverParams_::norm_traces] = P_.norm_traces;
//        tmpSP->params[SolverParams_::weights_ns] = P_.weights_ns;
        tmpSP->params[SolverParams_::tau_w] = P_.tau_w;
        tmpSP->params[SolverParams_::Tp_max_] = P_.Tp_max;
        tmpSP->params[SolverParams_::Td_max_] = P_.Td_max;
        tmpSP->params[SolverParams_::tau_ltp_] = P_.tau_ltp;
        tmpSP->params[SolverParams_::tau_ltd_] = P_.tau_ltd;
        tmpSP->params[SolverParams_::eta_ltp_] = P_.eta_ltp;
        tmpSP->params[SolverParams_::eta_ltd_] = P_.eta_ltd;
        tmpSP->params[SolverParams_::learn_rate] = P_.learn_rate;

        msg << "Processing steps (evolve dynamics)...\n";

        const double t_curspike_ = t_spike;
        double tp_runner = t_lastspike_;  // current (main) timepoint, updated stepwise
        double t_reward_on;
        double t_reward_off;
        double t_ref_on;
        double t_ref_off;

        while (abs(tp_runner - t_curspike_) > 0.01) {
            // the correct V_.__t_reward_idx is updated at the end of the loop, if needed.
            // always points to the correct, next valid reward period
            t_reward_on = P_.reward_times[V_.__t_reward_idx];
            t_reward_off = P_.reward_times[V_.__t_reward_idx] + P_.T_reward;
            t_ref_on = t_reward_off + nest::Time::get_resolution().get_ms();
            t_ref_off = t_ref_on + P_.T_tr;

            assert(t_reward_on && "Reward ON must be defined");
            assert(t_reward_off && "Reward OFF must be defined");
            msg << "State for synapse @" << this
                << "\n\tCURRENT TP: " << tp_runner
                << "\n\tt_reward_on / off: " << t_reward_on << " / " << t_reward_off
                << "\n\tt_refractory_on / off: " << t_ref_on << " / " << t_ref_off << "\n";

            // before reward onset
            if (tp_runner < t_reward_on)
            {
                _evolve_interval_block(tid, tp_runner, t_reward_on, t_curspike_, B_, tmpSP, false, false, msg);
            }

            // during reward window
            if (t_curspike_ >= t_reward_on && tp_runner < t_reward_off)
            {
                _evolve_interval_block(tid, tp_runner, t_reward_off, t_curspike_, B_, tmpSP, true, false, msg);
            }

            // during eligibility refractory window
            if (t_curspike_ >= t_ref_on && tp_runner < t_ref_off)
            {
                _evolve_interval_block(tid, tp_runner, t_ref_off, t_curspike_, B_, tmpSP, false, true, msg);
            }

            // after eligibility refractory window offset
            if (t_curspike_ > t_ref_off)
            {
                double next_reward_on = 1e12;
                // check if current spike is actually beyond the next reward zone
                if (check_and_advance_reward_counter(t_curspike_, msg)) {
                    next_reward_on = P_.reward_times[V_.__t_reward_idx];
                }

                _evolve_interval_block(tid, tp_runner, next_reward_on, t_curspike_, B_, tmpSP, false, false, msg);
            }
        }

//        LOG(M_DEBUG, "ShouvalConnectionMultiTh::evolve_dynamics_with_sampling()", msg.str());
        log_message(tid, msg);
    }


/**
* constructor
**/
    template < typename targetidentifierT >
    ShouvalConnectionMultiTh< targetidentifierT >::ShouvalConnectionMultiTh()
            :ConnectionBase()
    {
        P_.theta = 0.01; // estimate firing rate threshold
        P_.tau_w = 40.; // as ms
        P_.tau_ltp = 2000; // as ms
        P_.tau_ltd = 1000; // as ms
        P_.Tp_max = 0.95; // as real
        P_.Td_max = 1.0; // as real
        P_.eta_ltp = 1.;
        P_.eta_ltd = 0.55;
        P_.T_tr = 25; // as ms
        P_.T_reward = 25; // as ms
        P_.the_delay = 1; // as ms
        P_.learn_rate = 0.0045;
        P_.__gsl_error_tol = 1e-3;
        P_.sampling_interval = 0.;
        P_.weight_update_once = 0.;
//        P_.norm_ff = true;
        P_.norm_traces = 0.;
        P_.weights_ns = 1.;

        V_.__h = nest::Time::get_resolution().get_ms();
        V_.__t_reward_idx = 0;
        V_.unique_target = 0;
        V_.t_last_reward = -1.;  // TODO add comm
        V_.sampled_times.reserve(n_max_samples);
        V_.sampled_ltp.reserve(n_max_samples);
        V_.sampled_ltd.reserve(n_max_samples);
        V_.sampled_weights.reserve(n_max_samples);

        init_internals_block_symbols();

        for (unsigned i = 0; i < n_max_threads; ++i)
        {
            B_vec_[i] = new ODEStruct1_();
            init_buffers_(B_vec_[i]);
        }

        SP_.params[SolverParams_::tau_w] = P_.tau_w;
        SP_.params[SolverParams_::norm_traces] = P_.norm_traces;
//        SP_.params[SolverParams_::weights_ns] = P_.weights_ns;
        SP_.params[SolverParams_::Tp_max_] = P_.Tp_max;
        SP_.params[SolverParams_::Td_max_] = P_.Td_max;
        SP_.params[SolverParams_::tau_ltp_] = P_.tau_ltp;
        SP_.params[SolverParams_::tau_ltd_] = P_.tau_ltd;
        SP_.params[SolverParams_::eta_ltp_] = P_.eta_ltp;
        SP_.params[SolverParams_::eta_ltd_] = P_.eta_ltd;
        SP_.params[SolverParams_::reward_on] = 0;
        SP_.params[SolverParams_::refractory_on] = 0;
        SP_.params[SolverParams_::learn_rate] = P_.learn_rate;

        S_.y_[State_::rate_pre] = 0.0; // as real
        S_.y_[State_::rate_post] = 0.0; // as real
        S_.y_[State_::trace_ltd] = 0.0; // as real
        S_.y_[State_::trace_ltp] = 0.0; // as real
        S_.y_[State_::w] = 0.0; // as real

        t_lastspike_ = 0.;
        weight_ = 0.0;
    }

    template < typename targetidentifierT >
    ShouvalConnectionMultiTh< targetidentifierT >::ShouvalConnectionMultiTh( const ShouvalConnectionMultiTh< targetidentifierT >& rhs ):
            ConnectionBase(rhs), SP_(rhs.SP_)
            , B_vec_(rhs.B_vec_)
    {
        P_.theta = rhs.P_.theta;
        P_.tau_w = rhs.P_.tau_w;
        P_.tau_ltp = rhs.P_.tau_ltp;
        P_.tau_ltd = rhs.P_.tau_ltd;
        P_.Tp_max = rhs.P_.Tp_max;
        P_.Td_max = rhs.P_.Td_max;
        P_.eta_ltp = rhs.P_.eta_ltp;
        P_.eta_ltd = rhs.P_.eta_ltd;
        P_.T_tr = rhs.P_.T_tr;
        P_.T_reward = rhs.P_.T_reward;
        P_.the_delay = rhs.P_.the_delay;
        P_.learn_rate = rhs.P_.learn_rate;
        P_.reward_times = rhs.P_.reward_times;
        P_.sampling_interval = rhs.P_.sampling_interval;
        P_.norm_traces = rhs.P_.norm_traces;
        P_.weights_ns = rhs.P_.weights_ns;
        P_.weight_update_once = rhs.P_.weight_update_once;

        V_.__t_reward_idx = rhs.V_.__t_reward_idx;
        V_.t_last_reward = rhs.V_.t_last_reward;
        V_.sampled_times = rhs.V_.sampled_times;
        V_.sampled_ltp = rhs.V_.sampled_ltp;
        V_.sampled_ltd = rhs.V_.sampled_ltd;
        V_.sampled_weights = rhs.V_.sampled_weights;
        V_.sampled_hebbian = rhs.V_.sampled_hebbian;
        V_.unique_target = rhs.V_.unique_target;

        // state variables in ODE or kernel
        S_.y_[State_::w] = rhs.S_.y_[State_::w]; // as real
        S_.y_[State_::rate_pre] = rhs.S_.y_[State_::rate_pre]; // as real
        S_.y_[State_::rate_post] = rhs.S_.y_[State_::rate_post]; // as real
        S_.y_[State_::trace_ltp] = rhs.S_.y_[State_::trace_ltp]; // as real
        S_.y_[State_::trace_ltd] = rhs.S_.y_[State_::trace_ltd]; // as real

        t_lastspike_  = rhs.t_lastspike_;
        weight_ = rhs.weight_;
    }

    template<typename targetidentifierT>
    void
    ShouvalConnectionMultiTh<targetidentifierT>::get_status(DictionaryDatum &__d) const
    {
        ConnectionBase::get_status(__d);
        def<double>( __d, names::weight, weight_ );
        def<long>(__d, names::size_of, sizeof(*this));

        // parameters
        def<double>(__d, names::theta, get_theta());
        def<double>(__d, names::tau_w, get_tau_w());
        def<double>(__d, names::tau_ltp, get_tau_ltp());
        def<double>(__d, names::tau_ltd, get_tau_ltd());
        def<double>(__d, names::Tp_max, get_Tp_max());
        def<double>(__d, names::Td_max, get_Td_max());
        def<double>(__d, names::eta_ltp, get_eta_ltp());
        def<double>(__d, names::eta_ltd, get_eta_ltd());
        def<double>(__d, names::T_tr, get_T_tr());
        def<double>(__d, names::T_reward, get_T_reward());
        def<double>(__d, names::delay, P_.the_delay);
        def<double>(__d, names::learn_rate, get_learn_rate());
        def<double>(__d, names::weight_update_once, get_weight_update_once());
        def<double>(__d, names::norm_traces, get_norm_traces());
//        def<double>(__d, names::weights_ns, get_weights_ns());
        def<double>(__d, "weights_ns", get_weights_ns());

        def<std::vector<double>>(__d, names::reward_times, get_reward_times());
        def<std::vector<double>>(__d, names::sampled_times, get_sampled_times());
        def<std::vector<double>>(__d, names::sampled_ltp, get_sampled_ltp());
        def<std::vector<double>>(__d, names::sampled_ltd, get_sampled_ltd());
        def<std::vector<double>>(__d, names::sampled_weights, get_sampled_weights());
        def<std::vector<double>>(__d, names::sampled_hebbian, get_sampled_hebbian());
        def<double>(__d, names::sampling_interval, get_sampling_interval());
        def<double>(__d, "sample_cnt", V_.t_last_reward);

        // initial values for state variables in ODE or kernel
        def<double>(__d, names::w, get_w());
        def<double>(__d, names::trace_ltp, get_trace_ltp());
        def<double>(__d, names::trace_ltd, get_trace_ltd());
        def<double>(__d, names::rate_pre, get_rate_pre());
        def<double>(__d, names::rate_post, get_rate_post());
    }

    template<typename targetidentifierT>
    void
    ShouvalConnectionMultiTh<targetidentifierT>::set_status(const DictionaryDatum &__d, ConnectorModel &cm)
    {
        // parameters
        updateValue<double>(__d, names::theta, P_.theta );
        updateValue<double>(__d, names::tau_w, P_.tau_w );
        updateValue<double>(__d, names::tau_ltp, P_.tau_ltp);
        updateValue<double>(__d, names::tau_ltd, P_.tau_ltd);
        updateValue<double>(__d, names::Tp_max, P_.Tp_max);
        updateValue<double>(__d, names::Td_max, P_.Td_max);
        updateValue<double>(__d, names::eta_ltp, P_.eta_ltp);
        updateValue<double>(__d, names::eta_ltd, P_.eta_ltd);
        updateValue<double>(__d, names::T_tr, P_.T_tr);
        updateValue<double>(__d, names::T_reward, P_.T_reward);
        updateValue<double>(__d, names::the_delay, P_.the_delay);
        updateValue<double>(__d, names::learn_rate, P_.learn_rate);

        //! learning / integration options
        updateValue<double>(__d, names::weight_update_once, P_.weight_update_once);
        updateValue<double>(__d, names::norm_traces, P_.norm_traces);
//        updateValue<double>(__d, names::weights_ns, P_.weights_ns);
        updateValue<double>(__d, "weights_ns", P_.weights_ns);

        //! update reward times: we want to keep all previous times to avoid cross-batch inconsistencies where
        //! the last spike was before the last reward time in the previous batch and hence not yet processed.
        std::vector< double > tmp_reward_times;
        updateValue< std::vector< double > >(__d, names::reward_times, tmp_reward_times);

        //! add new reward times, if set
        for ( auto it : tmp_reward_times )
        {
            P_.reward_times.push_back(it);
        }

        updateValue<double>(__d, names::sampling_interval, P_.sampling_interval);
        if (not __d->lookup(names::sampling_interval).empty())
        {
            V_.t_last_reward = 0;
            V_.sampled_times.clear();
            V_.sampled_hebbian.clear();
            V_.sampled_weights.clear();
            V_.sampled_ltp.clear();
            V_.sampled_ltd.clear();
        }

        updateValue<double>(__d, names::w, S_.y_[State_::w]);
        updateValue<double>(__d, names::trace_ltp, S_.y_[State_::trace_ltp]);
        updateValue<double>(__d, names::trace_ltd, S_.y_[State_::trace_ltd]);
        updateValue<double>(__d, names::rate_pre, S_.y_[State_::rate_pre]);
        updateValue<double>(__d, names::rate_post, S_.y_[State_::rate_post]);

        ///////////////////////////
        // We now know that (ptmp, stmp) are consistent. We do not
        // write them back to (P_, S_) before we are also sure that
        // the properties to be set in the parent class are internally
        // consistent.
        ConnectionBase::set_status(__d, cm);

        SP_.params[SolverParams_::tau_w] = P_.tau_w;
        SP_.params[SolverParams_::tau_ltp_] = P_.tau_ltp;
        SP_.params[SolverParams_::tau_ltd_] = P_.tau_ltd;
        SP_.params[SolverParams_::Tp_max_] = P_.Tp_max;
        SP_.params[SolverParams_::Td_max_] = P_.Td_max;
        SP_.params[SolverParams_::learn_rate] = P_.learn_rate;

        //! DEPR: fake spike. fill recordings but do not change any member variables, only temporarily!
        //! NOW: fake spike. fill recordings AND DO change / reset member variables - see function desc!
        double t_fake_spike = -1;
        updateValue<double>(__d, names::fake_spike, t_fake_spike);
        if (t_fake_spike > 0. )
        {
//            //! if we're not sampling variables more densely but are only interested in their last value,
//            //! then we can ignore evolving the synapse dynamics if the last spike was sufficiently back in the past:
//            //! i.e., if the last spike has been processed through at least one reward window, then there are no
//            //! more contributions until the end.
//            if (P_.sampling_interval < 1 and
//                P_.reward_times.size() > 2 and
//                t_lastspike_ < P_.reward_times[P_.reward_times.size() - 2])
//            {
//                t_lastspike_ = t_fake_spike;  // ensure we'll continue from here next time!
//                weight_ = S_.y_[State_::w];
//            }
//            else
//            {
//                fill_missing_samples(t_fake_spike);
//            }
            fill_missing_samples(t_fake_spike);

            // recording structures will be cleared when sampling interval is changed
            // HOWEVER, we must reset all internal state variables (traces, rates) here to ensure that we start the
            // next batch with a clean slate!!!
            S_.y_[State_::trace_ltp] = 0.;
            S_.y_[State_::trace_ltd] = 0.;
            S_.y_[State_::rate_pre] = 0.;
            S_.y_[State_::rate_post] = 0.;
        }

        //! for direct control of the main weight_ variable.
        //! NOTE: for consistency reasons, when changing the weights both the S[w] and weight_ are set to the same value
        updateValue<double>(__d, names::weight, weight_);
    }

    template<typename targetidentifierT>
    ShouvalConnectionMultiTh<targetidentifierT>::ODEStruct1_::ODEStruct1_():
            __s( 0 ), __c( 0 ), __e( 0 )
    {
    }

    template<typename targetidentifierT>
    ShouvalConnectionMultiTh<targetidentifierT>::ODEStruct1_::ODEStruct1_(const ODEStruct1_ &ode)
    {
        __s = ode.__s;
        __c = ode.__c;
        __e = ode.__e;
        __sys = ode.__sys;
        __step = ode.__step;
        __integration_step = ode.__integration_step;
    }


    template < typename targetidentifierT >
    void ShouvalConnectionMultiTh< targetidentifierT >::reset_buffers_(ShouvalConnectionMultiTh::ODEStruct1_ *B_)
    {
        assert(B_->__c);
        assert(B_->__e);
        assert(B_->__s);
        assert(&B_->__sys);

        gsl_odeiv_step_reset( B_->__s );

        gsl_odeiv_evolve_reset( B_->__e );

        B_->__step = nest::Time::get_resolution().get_ms();
        B_->__integration_step = nest::Time::get_resolution().get_ms();
    }

    template < typename targetidentifierT >
    void ShouvalConnectionMultiTh< targetidentifierT >::init_buffers_(ShouvalConnectionMultiTh::ODEStruct1_ *B_)
    {
        if ( B_->__s == 0 )
        {
            B_->__s = gsl_odeiv_step_alloc( gsl_odeiv_step_rkf45, State_::STATE_VEC_SIZE );  // main
        }
        else
        {
            gsl_odeiv_step_reset( B_->__s );
        }

        if ( B_->__c == 0 )
        {
            B_->__c = gsl_odeiv_control_y_new( P_.__gsl_error_tol, 0.0 );
            assert(B_->__c);
        }
        else
        {
            gsl_odeiv_control_init( B_->__c, P_.__gsl_error_tol, 0.0, 1.0, 0.0 );
        }
        assert(B_->__c);


        if ( B_->__e == 0 )
        {
            B_->__e = gsl_odeiv_evolve_alloc( State_::STATE_VEC_SIZE );
        }
        else
        {
            gsl_odeiv_evolve_reset( B_->__e );
        }

        assert(B_->__c);
        assert(B_->__e);
        assert(B_->__s);
        assert(&B_->__sys);

        B_->__sys.function = ShouvalConnectionMultiThDynamics;
        B_->__sys.jacobian = NULL;
        B_->__sys.dimension = State_::STATE_VEC_SIZE;
//        B_->__sys.params = NULL; //reinterpret_cast< void* >( this );
        B_->__sys.params = reinterpret_cast< void* >( &SP_ );

        B_->__step = nest::Time::get_resolution().get_ms();
        B_->__integration_step = nest::Time::get_resolution().get_ms();
    }


    template < typename targetidentifierT >
    void ShouvalConnectionMultiTh< targetidentifierT >::init_internals_block_symbols()
    {
        V_.__P__rate_pre__rate_pre = std::exp((-V_.__h) / P_.tau_w);
        V_.__P__rate_post__rate_post = std::exp((-V_.__h) / P_.tau_w);
    }

    template < typename targetidentifierT >
    void ShouvalConnectionMultiTh< targetidentifierT >::reset_internal_state()
    {
        S_.y_[State_::rate_pre] = 0.;
        S_.y_[State_::rate_post] = 0.;
        S_.y_[State_::trace_ltp] = 0.;
        S_.y_[State_::trace_ltd] = 0.;
    }

    /**
     * Log the most important variables from the SP_, V_ and S_ variables.
     */
    template < typename targetidentifierT >
    void ShouvalConnectionMultiTh< targetidentifierT >::log_complete_state(std::string s)
    {
        std::ostringstream msg;
        msg << "[FAKE] Logging complete state: " << s;
        msg << "\n\t[FAKE] SP_"
            << "\n\t\tSP_.tau_w: " << SP_.tau_w
            << "\n\t\tSP_.reward_on: " << SP_.reward_on
            << "\n\t\tSP_.refractory_on: " << SP_.refractory_on;

        msg << "\n\t[FAKE] V_"
            << "\n\t\tV_.__t_reward_idx: " << V_.__t_reward_idx
            << "\n\t\tV_.__P__rate_pre__rate_pre: " << V_.__P__rate_pre__rate_pre
            << "\n\t\tV_.unique_target: " << V_.unique_target
            << "\n\t\tV_.sampled_times.size(): " << V_.sampled_times.size();

        msg << "\n\t[FAKE] S_"
            << "\n\t\tweight_: " << weight_
            << "\n\t\tS_.w: " << S_.y_[State_::w]
            << "\n\t\tS_.trace_ltp: " << S_.y_[State_::trace_ltp]
            << "\n\t\tS_.trace_ltd: " << S_.y_[State_::trace_ltd]
            << "\n\t\tS_.rate_pre: " << S_.y_[State_::rate_pre] << "\n\n";
        LOG(M_DEBUG, "[FAKE] ShouvalConnectionMultiTh::log_complete_state()", msg.str());
//        std::cout << msg.str();
    }

    /**
     * Simulates a fake incoming (presynaptic) spike in order to record all trace values up to the current
     * fake timepoint. Not particularly efficient, but it does the job for debugging / visualizing purposes.
     * @tparam targetidentifierT
     * @param t_fake_spike
     */
    template < typename targetidentifierT >
    void ShouvalConnectionMultiTh< targetidentifierT >::fill_missing_samples(const double t_fake_spike)
    {
        // can't use assert here, because some synapses may not have spiked at all, in which case the target is not set
//        assert(V_.unique_target && "Unique target has not been set!");
        if ( not V_.unique_target )
        {
            return;
        }

        const double dendritic_delay = get_delay();
        const thread tid = 0; // just pick one

        // need to save the current state to restore it after the fake spike
//        SolverParams_ copy_SP_ = SP_;
//        Variables_ copy_V_ = V_;
//        State_ copy_S_ = S_;
//        const double copy_t_lastspike_ = t_lastspike_;
//        log_complete_state("Saving state");

        B_vec_[tid]->__sys.params = reinterpret_cast< void* >( &SP_ );

        std::deque<histentry>::iterator start;
        std::deque<histentry>::iterator finish;

        V_.unique_target->get_history(t_lastspike_ - dendritic_delay, t_fake_spike - dendritic_delay, &start, &finish);

//        std::ostringstream msg;
//        msg << "######### [FAKE] post-synaptic processing t_lastspike_ " << t_lastspike_ << "##########\n";

        // facilitation due to post-synaptic spikes since last pre-synaptic spike
        while (start != finish) {
//            msg << "--------------------------------------------------\n"
//                << "[FAKE] start (" << start->t_ << ") != finish (" << finish->t_ << ")\n";

            const double old___h = V_.__h;
            const double t_curspike_ = start->t_;

            V_.__h = (start->t_ + dendritic_delay) - t_lastspike_;
            V_.__h -= 1.;
            init_internals_block_symbols();

            double rate_pre__tmp = V_.__P__rate_pre__rate_pre * S_.y_[State_::rate_pre];
            double rate_post__tmp = V_.__P__rate_post__rate_post * S_.y_[State_::rate_post];

//            LOG(M_DEBUG, "[FAKE] ShouvalConnectionMultiTh::send()", msg.str());
//            msg.str("");

            evolve_dynamics_X(t_curspike_, tid);

            /* replace analytically solvable variables with precisely integrated values  */
            S_.y_[State_::rate_pre] = rate_pre__tmp;
            S_.y_[State_::rate_post] = rate_post__tmp;

            V_.__h = old___h;
            init_internals_block_symbols();  // XXX: can be skipped?

            S_.y_[State_::rate_post] += 1.0 / SP_.params[ SolverParams_::tau_w ];

            t_lastspike_ = start->t_;
            ++start;
        }

        // update synapse internal state from `t_lastspike_` to `t_spike`
        const double old___h = V_.__h;

        V_.__h = t_fake_spike - t_lastspike_;

        if (V_.__h > 1E-9) {

            init_internals_block_symbols();
            double rate_pre__tmp = V_.__P__rate_pre__rate_pre * S_.y_[State_::rate_pre];
            double rate_post__tmp = V_.__P__rate_post__rate_post * S_.y_[State_::rate_post];


            evolve_dynamics_X(t_fake_spike, tid);

            /* replace analytically solvable variables with precisely integrated values  */
            S_.y_[State_::rate_pre] = rate_pre__tmp;
            S_.y_[State_::rate_post] = rate_post__tmp;
        }

        //! TODO we don't restore the state here to avoid computing this section twice. Since the state at the
        //! end of this function is still consistent, provided it is called for ALL connections and not just
        //! the recorded ones. In addition we reset the traces and all other state variables to 0 since we assume
        //! that this function is called at the end of a batch, and in the Matlab implementation all trials are
        //! independent and start with a blank state.
        t_lastspike_ = t_fake_spike;  // ensure we'll continue from here next time!
        reset_internal_state();

        //! if weight is only updated at the end of trial (fake spike)
        if (P_.weight_update_once > 0.5)
        {
            weight_ = S_.y_[State_::w];
        }
    }

    template<typename targetidentifierT>
    inline void
    ShouvalConnectionMultiTh<targetidentifierT>::send(Event &e, thread tid, const CommonSynapseProperties &cp)
    {
        Node *__target = get_target(tid);
        const double t_spike = e.get_stamp().get_ms();
        const double dendritic_delay = get_delay();

        if (!V_.unique_target)
        {
            V_.unique_target = __target;
        }
        else
        {
            // TODO this can be removed if certain
            assert(V_.unique_target == __target && "Unique target is different from previous one?!");
        }

        std::ostringstream msg;
        msg << "\n==============================\nProcessing spike @t: " << t_spike
            << "; synapse id(" << __target->get_gid() << ") address @: " << this << " in thread #" << tid
            << " with &SP_ @" << &SP_ << std::endl;

        // ensure each solver object points to the SP_ parameters of this synapse object,
        // otherwise the same SP_ object might be shared between multiple synapses (due to copy
        // constructors), which will lead to incorrect/conflicting handling of reward/refractory times
        B_vec_[tid]->__sys.params = reinterpret_cast< void* >( &SP_ );

        if (t_lastspike_ < 0.)
        {
            t_lastspike_ = 0.;  // this is the first preynaptic spike to be processed
        }

        // ///////////////////////////////////////////////////////////////////////
        // get spike history in relevant range (t1, t2] from post-synaptic neuron
        std::deque<histentry>::iterator start;
        std::deque<histentry>::iterator finish;

        __target->get_history(t_lastspike_ - dendritic_delay, t_spike - dendritic_delay, &start, &finish);

        msg << "######### post-synaptic processing ##########\n";

        /////////////////////////////////////////////////////////////////////////
        // facilitation due to post-synaptic spikes since last pre-synaptic spike
        /////////////////////////////////////////////////////////////////////////
        while (start != finish) {
            msg << "--------------------------------------------------\n"
                << "start (" << start->t_ << ") != finish (" << finish->t_ << ")\n";
            /**
             * update synapse internal state from `t_lastspike_` to `start->t_`
            **/
            const double old___h = V_.__h;
            const double t_curspike_ = start->t_;

            V_.__h = (start->t_ + dendritic_delay) - t_lastspike_;
            // if first post-synaptic spike, we need to correct 1 step
            {
                V_.__h -= 1.;
            }
            init_internals_block_symbols();

            double rate_pre__tmp = V_.__P__rate_pre__rate_pre * S_.y_[State_::rate_pre];
            double rate_post__tmp = V_.__P__rate_post__rate_post * S_.y_[State_::rate_post];

            log_message(tid, msg);
            evolve_dynamics_X(t_curspike_, tid);

            /* replace analytically solvable variables with precisely integrated values  */
            S_.y_[State_::rate_pre] = rate_pre__tmp;
            S_.y_[State_::rate_post] = rate_post__tmp;

            V_.__h = old___h;
            init_internals_block_symbols();  // XXX: can be skipped?

            S_.y_[State_::rate_post] += 1.0 / SP_.params[SolverParams_::tau_w];
            /**
             * internal state has now been fully updated to `start->t_ + dendritic_delay`
            **/
            t_lastspike_ = start->t_;
            msg << "internal state been fully updated to `start->t_ + dendritic_delay`"
                << "\n\t t_last_spike_: " << t_lastspike_
                << "\n";
            ++start;
        }
        msg << "################ [END] #################\n";
        msg << "######### pre-synaptic processing after all post-syn have been processed ##########\n";

        //! ///////////////////////////////////////////////////////////////////////
        /**
         * update synapse internal state from `t_lastspike_` to `t_spike`
        **/
        const double old___h = V_.__h;

        V_.__h = t_spike - t_lastspike_;
        msg << "[bar] V_.__h (pre) (t_spike - t_lastspike__): " << V_.__h << "\n";

        if (V_.__h > 1E-9) {
            msg << "EXTRA !! V_.__h > 1E-9, means we have to evolve " << V_.__h
                << " ms/steps till current tp.\n";

            init_internals_block_symbols();
            double rate_pre__tmp = V_.__P__rate_pre__rate_pre * S_.y_[State_::rate_pre];
            double rate_post__tmp = V_.__P__rate_post__rate_post * S_.y_[State_::rate_post];

            log_message(tid, msg);
            evolve_dynamics_X(t_spike, tid);

            /* replace analytically solvable variables with precisely integrated values  */
            S_.y_[State_::rate_pre] = rate_pre__tmp;
            S_.y_[State_::rate_post] = rate_post__tmp;
        }

        V_.__h = old___h;
        S_.y_[State_::rate_pre] += 1.0 / SP_.params[SolverParams_::tau_w];

        //! if weight is updated continuously, during each reward period and not just at the end of trial (fake spike),
        //! ensure that we use the most recently computed weight when emitting the current spike!
        if (P_.weight_update_once < 0.5)
        {
            weight_ = S_.y_[State_::w];
        }

        set_delay(P_.the_delay);
        const long __delay_steps = nest::Time::delay_ms_to_steps(get_delay());
        set_delay_steps(__delay_steps);
        e.set_receiver(*__target);
        if (P_.weights_ns > 0.5)
        {
            e.set_weight(weight_);
        }
        else
        {
            e.set_weight(weight_ * 1e3);  // scale spike weight to nS if weight is stored in microS
        }

        // use accessor functions (inherited from Connection< >) to obtain delay in steps and rport
        e.set_delay_steps( get_delay_steps() );
        e.set_rport( get_rport() );
        e();

        /**
         *  synapse internal state has now been fully updated to `t_spike`
        **/
        msg << "Synapse internal state has now been fully updated to `t_spike` " << t_spike << "ms with"
            << "\n\tS_.rate_pre (inc. +1/tau_w): " << S_.y_[State_::rate_pre]
            << "\n\tS_.rate_post: " << S_.y_[State_::rate_post]
            << "\n";

        t_lastspike_ = t_spike;
        log_message(tid, msg);
    }

} // namespace

#endif /* #ifndef SHOUVAL_CONNECTION_H */