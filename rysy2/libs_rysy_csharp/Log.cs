//------------------------------------------------------------------------------
// <auto-generated />
//
// This file was automatically generated by SWIG (http://www.swig.org).
// Version 3.0.12
//
// Do not make changes to this file unless you know what you are doing--modify
// the SWIG interface file instead.
//------------------------------------------------------------------------------


public class Log : global::System.IDisposable {
  private global::System.Runtime.InteropServices.HandleRef swigCPtr;
  protected bool swigCMemOwn;

  internal Log(global::System.IntPtr cPtr, bool cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = new global::System.Runtime.InteropServices.HandleRef(this, cPtr);
  }

  internal static global::System.Runtime.InteropServices.HandleRef getCPtr(Log obj) {
    return (obj == null) ? new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero) : obj.swigCPtr;
  }

  ~Log() {
    Dispose();
  }

  public virtual void Dispose() {
    lock(this) {
      if (swigCPtr.Handle != global::System.IntPtr.Zero) {
        if (swigCMemOwn) {
          swigCMemOwn = false;
          rysyPINVOKE.delete_Log(swigCPtr);
        }
        swigCPtr = new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero);
      }
      global::System.GC.SuppressFinalize(this);
    }
  }

  public Log() : this(rysyPINVOKE.new_Log__SWIG_0(), true) {
  }

  public Log(string file_name) : this(rysyPINVOKE.new_Log__SWIG_1(file_name), true) {
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public void set_output_file_name(string file_name) {
    rysyPINVOKE.Log_set_output_file_name(swigCPtr, file_name);
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public void put_string(string str) {
    rysyPINVOKE.Log_put_string(swigCPtr, str);
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

}
