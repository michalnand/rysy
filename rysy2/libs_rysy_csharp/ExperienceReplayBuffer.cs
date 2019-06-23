//------------------------------------------------------------------------------
// <auto-generated />
//
// This file was automatically generated by SWIG (http://www.swig.org).
// Version 3.0.12
//
// Do not make changes to this file unless you know what you are doing--modify
// the SWIG interface file instead.
//------------------------------------------------------------------------------


public class ExperienceReplayBuffer : global::System.IDisposable {
  private global::System.Runtime.InteropServices.HandleRef swigCPtr;
  protected bool swigCMemOwn;

  internal ExperienceReplayBuffer(global::System.IntPtr cPtr, bool cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = new global::System.Runtime.InteropServices.HandleRef(this, cPtr);
  }

  internal static global::System.Runtime.InteropServices.HandleRef getCPtr(ExperienceReplayBuffer obj) {
    return (obj == null) ? new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero) : obj.swigCPtr;
  }

  ~ExperienceReplayBuffer() {
    Dispose();
  }

  public virtual void Dispose() {
    lock(this) {
      if (swigCPtr.Handle != global::System.IntPtr.Zero) {
        if (swigCMemOwn) {
          swigCMemOwn = false;
          rysyPINVOKE.delete_ExperienceReplayBuffer(swigCPtr);
        }
        swigCPtr = new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero);
      }
      global::System.GC.SuppressFinalize(this);
    }
  }

  public ExperienceReplayBuffer() : this(rysyPINVOKE.new_ExperienceReplayBuffer__SWIG_0(), true) {
  }

  public ExperienceReplayBuffer(ExperienceReplayBuffer other) : this(rysyPINVOKE.new_ExperienceReplayBuffer__SWIG_1(ExperienceReplayBuffer.getCPtr(other)), true) {
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public ExperienceReplayBuffer(uint buffer_size, uint state_size, uint actions_count) : this(rysyPINVOKE.new_ExperienceReplayBuffer__SWIG_3(buffer_size, state_size, actions_count), true) {
  }

  public void init(uint buffer_size, uint state_size, uint actions_count) {
    rysyPINVOKE.ExperienceReplayBuffer_init(swigCPtr, buffer_size, state_size, actions_count);
  }

  public bool add(SWIGTYPE_p_std__vectorT_float_t state, SWIGTYPE_p_std__vectorT_float_t q_values, uint action, float reward, bool terminal) {
    bool ret = rysyPINVOKE.ExperienceReplayBuffer_add__SWIG_0(swigCPtr, SWIGTYPE_p_std__vectorT_float_t.getCPtr(state), SWIGTYPE_p_std__vectorT_float_t.getCPtr(q_values), action, reward, terminal);
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
    return ret;
  }

  public bool add(SWIGTYPE_p_std__vectorT_float_t state, SWIGTYPE_p_std__vectorT_float_t q_values, uint action, float reward) {
    bool ret = rysyPINVOKE.ExperienceReplayBuffer_add__SWIG_1(swigCPtr, SWIGTYPE_p_std__vectorT_float_t.getCPtr(state), SWIGTYPE_p_std__vectorT_float_t.getCPtr(q_values), action, reward);
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
    return ret;
  }

  public bool is_full() {
    bool ret = rysyPINVOKE.ExperienceReplayBuffer_is_full(swigCPtr);
    return ret;
  }

  public void compute(float gamma, float clamp_value) {
    rysyPINVOKE.ExperienceReplayBuffer_compute__SWIG_0(swigCPtr, gamma, clamp_value);
  }

  public void compute(float gamma) {
    rysyPINVOKE.ExperienceReplayBuffer_compute__SWIG_1(swigCPtr, gamma);
  }

  public void compute() {
    rysyPINVOKE.ExperienceReplayBuffer_compute__SWIG_2(swigCPtr);
  }

  public uint size() {
    uint ret = rysyPINVOKE.ExperienceReplayBuffer_size(swigCPtr);
    return ret;
  }

  public void print() {
    rysyPINVOKE.ExperienceReplayBuffer_print(swigCPtr);
  }

  public SWIGTYPE_p_std__vectorT_std__vectorT_float_t_t get_state() {
    SWIGTYPE_p_std__vectorT_std__vectorT_float_t_t ret = new SWIGTYPE_p_std__vectorT_std__vectorT_float_t_t(rysyPINVOKE.ExperienceReplayBuffer_get_state(swigCPtr), false);
    return ret;
  }

  public SWIGTYPE_p_std__vectorT_std__vectorT_float_t_t get_q_values() {
    SWIGTYPE_p_std__vectorT_std__vectorT_float_t_t ret = new SWIGTYPE_p_std__vectorT_std__vectorT_float_t_t(rysyPINVOKE.ExperienceReplayBuffer_get_q_values(swigCPtr), false);
    return ret;
  }

  public SWIGTYPE_p_std__vectorT_unsigned_int_t get_action() {
    SWIGTYPE_p_std__vectorT_unsigned_int_t ret = new SWIGTYPE_p_std__vectorT_unsigned_int_t(rysyPINVOKE.ExperienceReplayBuffer_get_action(swigCPtr), false);
    return ret;
  }

  public SWIGTYPE_p_std__vectorT_float_t get_reward() {
    SWIGTYPE_p_std__vectorT_float_t ret = new SWIGTYPE_p_std__vectorT_float_t(rysyPINVOKE.ExperienceReplayBuffer_get_reward(swigCPtr), false);
    return ret;
  }

  public SWIGTYPE_p_std__vectorT_bool_t get_terminal() {
    SWIGTYPE_p_std__vectorT_bool_t ret = new SWIGTYPE_p_std__vectorT_bool_t(rysyPINVOKE.ExperienceReplayBuffer_get_terminal(swigCPtr), false);
    return ret;
  }

}