#ifndef DataFormatsCommonSmartVector_H
#define DataFormatsCommonSmartVector_H

#include<vector>
#include<array>
#include<cstdint>


// a mimimal smart vector that can be either an array or a vector
// T must be an integer type (even if it may work with float as well)
template<typename T>
class SmartVector {
public :
  using Vector = std::vector<T>;
  static constexpr uint32_t maxSize = sizeof(Vector)/sizeof(T)-1;
  using Array = std::array<T,sizeof(Vector)/sizeof(T)>;
  union Variant {
    // all nop
    Variant(){}
    ~Variant(){}
    Variant(const Variant &){}
    Variant & operator=(const Variant &){ return *this;}
    Variant(Variant&&){}
    Variant & operator=(Variant &&){ return *this;}

    Array a;
    Vector v;
  };

  SmartVector(){
    m_container.a.back()=0;
  }

  ~SmartVector(){
   if(!m_isArray)
     m_container.v.~Vector();
  }

  SmartVector(const SmartVector& sm) : m_isArray(sm.m_isArray) {
   if(m_isArray) {
    m_container.a=sm.m_container.a;
   } else {
    std::fill(m_container.a.begin(), m_container.a.end(),0);
    m_container.v=sm.m_container.v;
   }
  }
  SmartVector& operator=(const SmartVector& sm) {
   if(!m_isArray) m_container.v.~Vector();
   if(sm.m_isArray) {
    m_container.a=sm.m_container.a;
   } else {
    std::fill(m_container.a.begin(), m_container.a.end(),0);
    m_container.v=sm.m_container.v;
   }
   m_isArray = sm.m_isArray;
   return *this;
  }

  SmartVector(SmartVector&& sm) : m_isArray(sm.m_isArray) {
   if(m_isArray) {
    m_container.a=std::move(sm.m_container.a);
   } else {
    std::fill(m_container.a.begin(), m_container.a.end(),0);
    m_container.v=std::move(sm.m_container.v);
   }
  }

  SmartVector& operator=(SmartVector&& sm) {
   if(!m_isArray) m_container.v.~Vector();
   if(sm.m_isArray) {
    m_container.a=std::move(sm.m_container.a);
   } else {
    if(m_isArray) std::fill(m_container.a.begin(), m_container.a.end(),0);
    m_container.v=std::move(sm.m_container.v);
   }
   m_isArray = sm.m_isArray;
   return *this;
  }


  template<typename Iter>
  SmartVector(Iter b, Iter e) {
     initialize(b,e);
  }


  void initialize(uint32_t size) {
    if(!m_isArray) m_container.v.~Vector();
    if (size<=maxSize) {
      m_isArray=true;
      m_container.a.back()=size;
    } else {
       m_isArray=false;
       std::fill(m_container.a.begin(), m_container.a.end(),0);
       m_container.v.resize(size);
    }
  }

  template<typename Iter>
  void initialize(Iter b, Iter e) {
    if(!m_isArray) m_container.v.~Vector();
    if (e-b<=maxSize) {
       m_isArray=true;
       auto & a = m_container.a;
       std::copy(b,e,a.begin());
       a.back()=e-b;
    } else {
       m_isArray=false;
       std::fill(m_container.a.begin(), m_container.a.end(),0);
       m_container.v.insert(m_container.v.end(),b,e);
    }
  }

  template<typename Iter>
  void extend(Iter b, Iter e) {
    if(m_isArray) {
      auto & a = m_container.a;
      auto cs = a.back();
      uint32_t ns = (e-b)+cs;
      if (ns<=maxSize) {
        std::copy(b,e,&a[cs]);
        a.back()=ns;
      } else {
        Vector v; v.reserve(ns);
        v.insert(v.end(),m_container.a.begin(),m_container.a.begin()+cs);
        v.insert(v.end(),b,e);
        std::fill(m_container.a.begin(), m_container.a.end(),0);
        m_container.v = std::move(v);
        m_isArray=false;
      }
    }else {
     m_container.v.insert(m_container.v.end(),b,e);
    }
  }


  T const * data() const {
    if(m_isArray)
       return m_container.a.data();
    else
       return m_container.v.data();
  }

  T * data()  {
    if(m_isArray)
       return m_container.a.data();
    else
       return m_container.v.data();
  }

  T const * begin() const {
    return data();
  }

  T const * end() const {
    if(m_isArray)
      return m_container.a.data() + m_container.a.back();
    else
      return  m_container.v.data() + m_container.v.size();
  }

  T const & operator[](uint32_t i) const {
    return *(begin()+i);
  }

  uint32_t size() const {
    if(m_isArray)
       return m_container.a.back();
    else
       return m_container.v.size();
  }


  bool isArray() const { return m_isArray;}
private:
  Variant m_container;
  bool m_isArray = true;
};
#endif
