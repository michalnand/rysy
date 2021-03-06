//------------------------------------------------------------------------------
// <auto-generated />
//
// This file was automatically generated by SWIG (http://www.swig.org).
// Version 3.0.12
//
// Do not make changes to this file unless you know what you are doing--modify
// the SWIG interface file instead.
//------------------------------------------------------------------------------


public class DQNA : global::System.IDisposable {
  private global::System.Runtime.InteropServices.HandleRef swigCPtr;
  protected bool swigCMemOwn;

  internal DQNA(global::System.IntPtr cPtr, bool cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = new global::System.Runtime.InteropServices.HandleRef(this, cPtr);
  }

  internal static global::System.Runtime.InteropServices.HandleRef getCPtr(DQNA obj) {
    return (obj == null) ? new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero) : obj.swigCPtr;
  }

  ~DQNA() {
    Dispose();
  }

  public virtual void Dispose() {
    lock(this) {
      if (swigCPtr.Handle != global::System.IntPtr.Zero) {
        if (swigCMemOwn) {
          swigCMemOwn = false;
          rysyPINVOKE.delete_DQNA(swigCPtr);
        }
        swigCPtr = new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero);
      }
      global::System.GC.SuppressFinalize(this);
    }
  }

  public DQNA() : this(rysyPINVOKE.new_DQNA__SWIG_0(), true) {
  }

  public DQNA(Shape state_shape, uint actions_count, float gamma, uint replay_buffer_size, string features_network_config_file_name, string reconstruction_network_config_file_name, string q_network_config_file_name) : this(rysyPINVOKE.new_DQNA__SWIG_1(Shape.getCPtr(state_shape), actions_count, gamma, replay_buffer_size, features_network_config_file_name, reconstruction_network_config_file_name, q_network_config_file_name), true) {
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public DQNA(DQNA other) : this(rysyPINVOKE.new_DQNA__SWIG_2(DQNA.getCPtr(other)), true) {
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public void init(Shape state_shape, uint actions_count, float gamma, uint replay_buffer_size, string features_network_config_file_name, string reconstruction_network_config_file_name, string q_network_config_file_name) {
    rysyPINVOKE.DQNA_init(swigCPtr, Shape.getCPtr(state_shape), actions_count, gamma, replay_buffer_size, features_network_config_file_name, reconstruction_network_config_file_name, q_network_config_file_name);
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public SWIGTYPE_p_std__vectorT_float_t forward(SWIGTYPE_p_std__vectorT_float_t state) {
    SWIGTYPE_p_std__vectorT_float_t ret = new SWIGTYPE_p_std__vectorT_float_t(rysyPINVOKE.DQNA_forward(swigCPtr, SWIGTYPE_p_std__vectorT_float_t.getCPtr(state)), false);
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
    return ret;
  }

  public SWIGTYPE_p_std__vectorT_float_t get_q_values() {
    SWIGTYPE_p_std__vectorT_float_t ret = new SWIGTYPE_p_std__vectorT_float_t(rysyPINVOKE.DQNA_get_q_values(swigCPtr), false);
    return ret;
  }

  public bool add(SWIGTYPE_p_std__vectorT_float_t state, SWIGTYPE_p_std__vectorT_float_t q_values, uint action, float reward, bool terminal) {
    bool ret = rysyPINVOKE.DQNA_add__SWIG_0(swigCPtr, SWIGTYPE_p_std__vectorT_float_t.getCPtr(state), SWIGTYPE_p_std__vectorT_float_t.getCPtr(q_values), action, reward, terminal);
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
    return ret;
  }

  public bool add(SWIGTYPE_p_std__vectorT_float_t state, SWIGTYPE_p_std__vectorT_float_t q_values, uint action, float reward) {
    bool ret = rysyPINVOKE.DQNA_add__SWIG_1(swigCPtr, SWIGTYPE_p_std__vectorT_float_t.getCPtr(state), SWIGTYPE_p_std__vectorT_float_t.getCPtr(q_values), action, reward);
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
    return ret;
  }

  public bool is_full() {
    bool ret = rysyPINVOKE.DQNA_is_full(swigCPtr);
    return ret;
  }

  public void train() {
    rysyPINVOKE.DQNA_train(swigCPtr);
  }

  public void print() {
    rysyPINVOKE.DQNA_print(swigCPtr);
  }

  public void print_buffer() {
    rysyPINVOKE.DQNA_print_buffer(swigCPtr);
  }

  public void save(string path) {
    rysyPINVOKE.DQNA_save(swigCPtr, path);
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public void load_weights(string file_name_prefix) {
    rysyPINVOKE.DQNA_load_weights(swigCPtr, file_name_prefix);
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

}
