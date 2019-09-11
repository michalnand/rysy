//------------------------------------------------------------------------------
// <auto-generated />
//
// This file was automatically generated by SWIG (http://www.swig.org).
// Version 3.0.12
//
// Do not make changes to this file unless you know what you are doing--modify
// the SWIG interface file instead.
//------------------------------------------------------------------------------


public class ImageSave : global::System.IDisposable {
  private global::System.Runtime.InteropServices.HandleRef swigCPtr;
  protected bool swigCMemOwn;

  internal ImageSave(global::System.IntPtr cPtr, bool cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = new global::System.Runtime.InteropServices.HandleRef(this, cPtr);
  }

  internal static global::System.Runtime.InteropServices.HandleRef getCPtr(ImageSave obj) {
    return (obj == null) ? new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero) : obj.swigCPtr;
  }

  ~ImageSave() {
    Dispose();
  }

  public virtual void Dispose() {
    lock(this) {
      if (swigCPtr.Handle != global::System.IntPtr.Zero) {
        if (swigCMemOwn) {
          swigCMemOwn = false;
          rysyPINVOKE.delete_ImageSave(swigCPtr);
        }
        swigCPtr = new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero);
      }
      global::System.GC.SuppressFinalize(this);
    }
  }

  public ImageSave(uint width, uint height, bool grayscale, bool display_enabled) : this(rysyPINVOKE.new_ImageSave__SWIG_0(width, height, grayscale, display_enabled), true) {
  }

  public ImageSave(uint width, uint height, bool grayscale) : this(rysyPINVOKE.new_ImageSave__SWIG_1(width, height, grayscale), true) {
  }

  public void save(string file_name, SWIGTYPE_p_std__vectorT_float_t v) {
    rysyPINVOKE.ImageSave_save__SWIG_0(swigCPtr, file_name, SWIGTYPE_p_std__vectorT_float_t.getCPtr(v));
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public void save(string file_name, SWIGTYPE_p_std__vectorT_std__vectorT_float_t_t v) {
    rysyPINVOKE.ImageSave_save__SWIG_1(swigCPtr, file_name, SWIGTYPE_p_std__vectorT_std__vectorT_float_t_t.getCPtr(v));
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public void save(string file_name, SWIGTYPE_p_std__vectorT_std__vectorT_std__vectorT_float_t_t_t v) {
    rysyPINVOKE.ImageSave_save__SWIG_2(swigCPtr, file_name, SWIGTYPE_p_std__vectorT_std__vectorT_std__vectorT_float_t_t_t.getCPtr(v));
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public void show(SWIGTYPE_p_std__vectorT_float_t v, string window_name) {
    rysyPINVOKE.ImageSave_show__SWIG_0(swigCPtr, SWIGTYPE_p_std__vectorT_float_t.getCPtr(v), window_name);
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public void show(SWIGTYPE_p_std__vectorT_float_t v) {
    rysyPINVOKE.ImageSave_show__SWIG_1(swigCPtr, SWIGTYPE_p_std__vectorT_float_t.getCPtr(v));
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

}
