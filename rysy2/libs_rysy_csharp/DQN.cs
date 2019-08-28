//------------------------------------------------------------------------------
// <auto-generated />
//
// This file was automatically generated by SWIG (http://www.swig.org).
// Version 3.0.12
//
// Do not make changes to this file unless you know what you are doing--modify
// the SWIG interface file instead.
//------------------------------------------------------------------------------


public class DQN : global::System.IDisposable {
  private global::System.Runtime.InteropServices.HandleRef swigCPtr;
  protected bool swigCMemOwn;

  internal DQN(global::System.IntPtr cPtr, bool cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = new global::System.Runtime.InteropServices.HandleRef(this, cPtr);
  }

  internal static global::System.Runtime.InteropServices.HandleRef getCPtr(DQN obj) {
    return (obj == null) ? new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero) : obj.swigCPtr;
  }

  ~DQN() {
    Dispose();
  }

  public virtual void Dispose() {
    lock(this) {
      if (swigCPtr.Handle != global::System.IntPtr.Zero) {
        if (swigCMemOwn) {
          swigCMemOwn = false;
          rysyPINVOKE.delete_DQN(swigCPtr);
        }
        swigCPtr = new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero);
      }
      global::System.GC.SuppressFinalize(this);
    }
  }

  public DQN() : this(rysyPINVOKE.new_DQN__SWIG_0(), true) {
  }

  public DQN(Shape state_shape, uint actions_count, float gamma, uint replay_buffer_size, string network_config_file_name) : this(rysyPINVOKE.new_DQN__SWIG_1(Shape.getCPtr(state_shape), actions_count, gamma, replay_buffer_size, network_config_file_name), true) {
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public DQN(Shape state_shape, uint actions_count, float gamma, uint replay_buffer_size) : this(rysyPINVOKE.new_DQN__SWIG_2(Shape.getCPtr(state_shape), actions_count, gamma, replay_buffer_size), true) {
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public DQN(Shape state_shape, uint actions_count, float gamma) : this(rysyPINVOKE.new_DQN__SWIG_3(Shape.getCPtr(state_shape), actions_count, gamma), true) {
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public DQN(Shape state_shape, uint actions_count) : this(rysyPINVOKE.new_DQN__SWIG_4(Shape.getCPtr(state_shape), actions_count), true) {
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public DQN(DQN other) : this(rysyPINVOKE.new_DQN__SWIG_5(DQN.getCPtr(other)), true) {
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public void init(Shape state_shape, uint actions_count, float gamma, uint replay_buffer_size, string network_config_file_name) {
    rysyPINVOKE.DQN_init__SWIG_0(swigCPtr, Shape.getCPtr(state_shape), actions_count, gamma, replay_buffer_size, network_config_file_name);
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public void init(Shape state_shape, uint actions_count, float gamma, uint replay_buffer_size) {
    rysyPINVOKE.DQN_init__SWIG_1(swigCPtr, Shape.getCPtr(state_shape), actions_count, gamma, replay_buffer_size);
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public void init(Shape state_shape, uint actions_count, float gamma) {
    rysyPINVOKE.DQN_init__SWIG_2(swigCPtr, Shape.getCPtr(state_shape), actions_count, gamma);
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public void init(Shape state_shape, uint actions_count) {
    rysyPINVOKE.DQN_init__SWIG_3(swigCPtr, Shape.getCPtr(state_shape), actions_count);
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public void add_layer(string layer_type, Shape shape) {
    rysyPINVOKE.DQN_add_layer__SWIG_0(swigCPtr, layer_type, Shape.getCPtr(shape));
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public void add_layer(string layer_type) {
    rysyPINVOKE.DQN_add_layer__SWIG_1(swigCPtr, layer_type);
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public SWIGTYPE_p_std__vectorT_float_t forward(SWIGTYPE_p_std__vectorT_float_t state) {
    SWIGTYPE_p_std__vectorT_float_t ret = new SWIGTYPE_p_std__vectorT_float_t(rysyPINVOKE.DQN_forward__SWIG_0(swigCPtr, SWIGTYPE_p_std__vectorT_float_t.getCPtr(state)), false);
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
    return ret;
  }

  public SWIGTYPE_p_std__vectorT_float_t forward(SWIGTYPE_p_float state) {
    SWIGTYPE_p_std__vectorT_float_t ret = new SWIGTYPE_p_std__vectorT_float_t(rysyPINVOKE.DQN_forward__SWIG_1(swigCPtr, SWIGTYPE_p_float.getCPtr(state)), false);
    return ret;
  }

  public SWIGTYPE_p_std__vectorT_float_t get_q_values() {
    SWIGTYPE_p_std__vectorT_float_t ret = new SWIGTYPE_p_std__vectorT_float_t(rysyPINVOKE.DQN_get_q_values(swigCPtr), false);
    return ret;
  }

  public bool add(SWIGTYPE_p_std__vectorT_float_t state, SWIGTYPE_p_std__vectorT_float_t q_values, uint action, float reward, bool terminal) {
    bool ret = rysyPINVOKE.DQN_add__SWIG_0(swigCPtr, SWIGTYPE_p_std__vectorT_float_t.getCPtr(state), SWIGTYPE_p_std__vectorT_float_t.getCPtr(q_values), action, reward, terminal);
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
    return ret;
  }

  public bool add(SWIGTYPE_p_std__vectorT_float_t state, SWIGTYPE_p_std__vectorT_float_t q_values, uint action, float reward) {
    bool ret = rysyPINVOKE.DQN_add__SWIG_1(swigCPtr, SWIGTYPE_p_std__vectorT_float_t.getCPtr(state), SWIGTYPE_p_std__vectorT_float_t.getCPtr(q_values), action, reward);
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
    return ret;
  }

  public bool add(SWIGTYPE_p_float state, SWIGTYPE_p_std__vectorT_float_t q_values, uint action, float reward, bool terminal) {
    bool ret = rysyPINVOKE.DQN_add__SWIG_2(swigCPtr, SWIGTYPE_p_float.getCPtr(state), SWIGTYPE_p_std__vectorT_float_t.getCPtr(q_values), action, reward, terminal);
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
    return ret;
  }

  public bool add(SWIGTYPE_p_float state, SWIGTYPE_p_std__vectorT_float_t q_values, uint action, float reward) {
    bool ret = rysyPINVOKE.DQN_add__SWIG_3(swigCPtr, SWIGTYPE_p_float.getCPtr(state), SWIGTYPE_p_std__vectorT_float_t.getCPtr(q_values), action, reward);
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
    return ret;
  }

  public bool is_full() {
    bool ret = rysyPINVOKE.DQN_is_full(swigCPtr);
    return ret;
  }

  public void train() {
    rysyPINVOKE.DQN_train(swigCPtr);
  }

  public void print() {
    rysyPINVOKE.DQN_print(swigCPtr);
  }

  public void print_buffer() {
    rysyPINVOKE.DQN_print_buffer(swigCPtr);
  }

  public void save(string path) {
    rysyPINVOKE.DQN_save(swigCPtr, path);
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public void load_weights(string file_name_prefix) {
    rysyPINVOKE.DQN_load_weights(swigCPtr, file_name_prefix);
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public void kernel_visualisation(string path) {
    rysyPINVOKE.DQN_kernel_visualisation(swigCPtr, path);
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public void activity_visualisation(string path, SWIGTYPE_p_std__vectorT_float_t state) {
    rysyPINVOKE.DQN_activity_visualisation(swigCPtr, path, SWIGTYPE_p_std__vectorT_float_t.getCPtr(state));
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

}
