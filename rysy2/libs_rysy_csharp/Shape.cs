//------------------------------------------------------------------------------
// <auto-generated />
//
// This file was automatically generated by SWIG (http://www.swig.org).
// Version 3.0.12
//
// Do not make changes to this file unless you know what you are doing--modify
// the SWIG interface file instead.
//------------------------------------------------------------------------------


public class Shape : global::System.IDisposable {
  private global::System.Runtime.InteropServices.HandleRef swigCPtr;
  protected bool swigCMemOwn;

  internal Shape(global::System.IntPtr cPtr, bool cMemoryOwn) {
    swigCMemOwn = cMemoryOwn;
    swigCPtr = new global::System.Runtime.InteropServices.HandleRef(this, cPtr);
  }

  internal static global::System.Runtime.InteropServices.HandleRef getCPtr(Shape obj) {
    return (obj == null) ? new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero) : obj.swigCPtr;
  }

  ~Shape() {
    Dispose();
  }

  public virtual void Dispose() {
    lock(this) {
      if (swigCPtr.Handle != global::System.IntPtr.Zero) {
        if (swigCMemOwn) {
          swigCMemOwn = false;
          rysyPINVOKE.delete_Shape(swigCPtr);
        }
        swigCPtr = new global::System.Runtime.InteropServices.HandleRef(null, global::System.IntPtr.Zero);
      }
      global::System.GC.SuppressFinalize(this);
    }
  }

  public Shape() : this(rysyPINVOKE.new_Shape__SWIG_0(), true) {
  }

  public Shape(Shape other) : this(rysyPINVOKE.new_Shape__SWIG_1(Shape.getCPtr(other)), true) {
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public Shape(sShape shape) : this(rysyPINVOKE.new_Shape__SWIG_3(sShape.getCPtr(shape)), true) {
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public Shape(uint width, uint height, uint depth, uint time) : this(rysyPINVOKE.new_Shape__SWIG_4(width, height, depth, time), true) {
  }

  public Shape(uint width, uint height, uint depth) : this(rysyPINVOKE.new_Shape__SWIG_5(width, height, depth), true) {
  }

  public Shape(uint width, uint height) : this(rysyPINVOKE.new_Shape__SWIG_6(width, height), true) {
  }

  public Shape(uint width) : this(rysyPINVOKE.new_Shape__SWIG_7(width), true) {
  }

  public uint w() {
    uint ret = rysyPINVOKE.Shape_w(swigCPtr);
    return ret;
  }

  public uint h() {
    uint ret = rysyPINVOKE.Shape_h(swigCPtr);
    return ret;
  }

  public uint d() {
    uint ret = rysyPINVOKE.Shape_d(swigCPtr);
    return ret;
  }

  public uint t() {
    uint ret = rysyPINVOKE.Shape_t(swigCPtr);
    return ret;
  }

  public sShape get() {
    sShape ret = new sShape(rysyPINVOKE.Shape_get(swigCPtr), true);
    return ret;
  }

  public void set(sShape shape) {
    rysyPINVOKE.Shape_set__SWIG_0(swigCPtr, sShape.getCPtr(shape));
    if (rysyPINVOKE.SWIGPendingException.Pending) throw rysyPINVOKE.SWIGPendingException.Retrieve();
  }

  public void set(uint width, uint height, uint depth, uint time) {
    rysyPINVOKE.Shape_set__SWIG_1(swigCPtr, width, height, depth, time);
  }

  public void set(uint width, uint height, uint depth) {
    rysyPINVOKE.Shape_set__SWIG_2(swigCPtr, width, height, depth);
  }

  public void set(uint width, uint height) {
    rysyPINVOKE.Shape_set__SWIG_3(swigCPtr, width, height);
  }

  public void set(uint width) {
    rysyPINVOKE.Shape_set__SWIG_4(swigCPtr, width);
  }

  public uint size() {
    uint ret = rysyPINVOKE.Shape_size(swigCPtr);
    return ret;
  }

  public void print() {
    rysyPINVOKE.Shape_print(swigCPtr);
  }

}
