//------------------------------------------------------------------------------
// <auto-generated />
//
// This file was automatically generated by SWIG (http://www.swig.org).
// Version 3.0.12
//
// Do not make changes to this file unless you know what you are doing--modify
// the SWIG interface file instead.
//------------------------------------------------------------------------------


public class DatasetInterface : global::System.IDisposable {
  private global::System.Runtime.InteropServices.HandleRef swigCPtr;
  protected bool swigCMemOwn;

  internal DatasetInterface(global::System.IntPtr cPtr, bool cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = new global::System.Runtime.InteropServices.HandleRef(this, cPtr);
  }

  internal static global::System.Runtime.InteropServices.HandleRef getCPtr(DatasetInterface obj) {
    return (obj == null) ? new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero) : obj.swigCPtr;
  }

  ~DatasetInterface() {
    Dispose();
  }

  public virtual void Dispose() {
    lock(this) {
      if (swigCPtr.Handle != global::System.IntPtr.Zero) {
        if (swigCMemOwn) {
          swigCMemOwn = false;
          rysyPINVOKE.delete_DatasetInterface(swigCPtr);
        }
        swigCPtr = new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero);
      }
      global::System.GC.SuppressFinalize(this);
    }
  }

  public DatasetInterface() : this(rysyPINVOKE.new_DatasetInterface__SWIG_0(), true) {
  }

  public DatasetInterface(DatasetInterface other) : this(rysyPINVOKE.new_DatasetInterface__SWIG_1(DatasetInterface.getCPtr(other)), true) {
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public DatasetInterface(Shape input_shape, Shape output_shape) : this(rysyPINVOKE.new_DatasetInterface__SWIG_3(Shape.getCPtr(input_shape), Shape.getCPtr(output_shape)), true) {
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public Shape get_input_shape() {
    Shape ret = new Shape(rysyPINVOKE.DatasetInterface_get_input_shape(swigCPtr), true);
    return ret;
  }

  public Shape get_output_shape() {
    Shape ret = new Shape(rysyPINVOKE.DatasetInterface_get_output_shape(swigCPtr), true);
    return ret;
  }

  public uint get_training_count() {
    uint ret = rysyPINVOKE.DatasetInterface_get_training_count(swigCPtr);
    return ret;
  }

  public uint get_testing_count() {
    uint ret = rysyPINVOKE.DatasetInterface_get_testing_count(swigCPtr);
    return ret;
  }

  public uint get_classes_count() {
    uint ret = rysyPINVOKE.DatasetInterface_get_classes_count(swigCPtr);
    return ret;
  }

  public virtual void set_training_idx(uint idx) {
    rysyPINVOKE.DatasetInterface_set_training_idx(swigCPtr, idx);
  }

  public virtual void set_random_training_idx() {
    rysyPINVOKE.DatasetInterface_set_random_training_idx(swigCPtr);
  }

  public virtual SWIGTYPE_p_std__vectorT_float_t get_training_input() {
    SWIGTYPE_p_std__vectorT_float_t ret = new SWIGTYPE_p_std__vectorT_float_t(rysyPINVOKE.DatasetInterface_get_training_input(swigCPtr), false);
    return ret;
  }

  public virtual SWIGTYPE_p_std__vectorT_float_t get_training_output() {
    SWIGTYPE_p_std__vectorT_float_t ret = new SWIGTYPE_p_std__vectorT_float_t(rysyPINVOKE.DatasetInterface_get_training_output(swigCPtr), false);
    return ret;
  }

  public virtual SWIGTYPE_p_std__vectorT_std__vectorT_float_t_t get_training_input_all() {
    SWIGTYPE_p_std__vectorT_std__vectorT_float_t_t ret = new SWIGTYPE_p_std__vectorT_std__vectorT_float_t_t(rysyPINVOKE.DatasetInterface_get_training_input_all(swigCPtr), false);
    return ret;
  }

  public virtual SWIGTYPE_p_std__vectorT_std__vectorT_float_t_t get_training_output_all() {
    SWIGTYPE_p_std__vectorT_std__vectorT_float_t_t ret = new SWIGTYPE_p_std__vectorT_std__vectorT_float_t_t(rysyPINVOKE.DatasetInterface_get_training_output_all(swigCPtr), false);
    return ret;
  }

  public virtual SWIGTYPE_p_std__vectorT_float_t get_testing_input(uint idx) {
    SWIGTYPE_p_std__vectorT_float_t ret = new SWIGTYPE_p_std__vectorT_float_t(rysyPINVOKE.DatasetInterface_get_testing_input(swigCPtr, idx), false);
    return ret;
  }

  public virtual SWIGTYPE_p_std__vectorT_float_t get_testing_output(uint idx) {
    SWIGTYPE_p_std__vectorT_float_t ret = new SWIGTYPE_p_std__vectorT_float_t(rysyPINVOKE.DatasetInterface_get_testing_output(swigCPtr, idx), false);
    return ret;
  }

  public virtual SWIGTYPE_p_std__vectorT_std__vectorT_float_t_t get_testing_input_all() {
    SWIGTYPE_p_std__vectorT_std__vectorT_float_t_t ret = new SWIGTYPE_p_std__vectorT_std__vectorT_float_t_t(rysyPINVOKE.DatasetInterface_get_testing_input_all(swigCPtr), false);
    return ret;
  }

  public virtual SWIGTYPE_p_std__vectorT_std__vectorT_float_t_t get_testing_output_all() {
    SWIGTYPE_p_std__vectorT_std__vectorT_float_t_t ret = new SWIGTYPE_p_std__vectorT_std__vectorT_float_t_t(rysyPINVOKE.DatasetInterface_get_testing_output_all(swigCPtr), false);
    return ret;
  }

  public void set_input_shape(Shape input_shape) {
    rysyPINVOKE.DatasetInterface_set_input_shape(swigCPtr, Shape.getCPtr(input_shape));
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public void set_output_shape(Shape output_shape) {
    rysyPINVOKE.DatasetInterface_set_output_shape(swigCPtr, Shape.getCPtr(output_shape));
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public void add_training(SWIGTYPE_p_std__vectorT_float_t input, SWIGTYPE_p_std__vectorT_float_t output) {
    rysyPINVOKE.DatasetInterface_add_training(swigCPtr, SWIGTYPE_p_std__vectorT_float_t.getCPtr(input), SWIGTYPE_p_std__vectorT_float_t.getCPtr(output));
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public void add_testing(SWIGTYPE_p_std__vectorT_float_t input, SWIGTYPE_p_std__vectorT_float_t output) {
    rysyPINVOKE.DatasetInterface_add_testing(swigCPtr, SWIGTYPE_p_std__vectorT_float_t.getCPtr(input), SWIGTYPE_p_std__vectorT_float_t.getCPtr(output));
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public void print() {
    rysyPINVOKE.DatasetInterface_print(swigCPtr);
  }

  public void clear() {
    rysyPINVOKE.DatasetInterface_clear(swigCPtr);
  }

  public void normalise_input() {
    rysyPINVOKE.DatasetInterface_normalise_input(swigCPtr);
  }

  public void normalise_output() {
    rysyPINVOKE.DatasetInterface_normalise_output(swigCPtr);
  }

}
