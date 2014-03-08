#ifndef DataFormats_Common_DetSetNew_h
#define DataFormats_Common_DetSetNew_h



#include "FWCore/Utilities/interface/GCC11Compatibility.h"

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
#define USE_ATOMIC
// #warning using atomic
#endif

#include <vector>
#include <memory>
#include <cassert>


namespace edmNew {
  //  FIXME move it elsewhere....
  typedef unsigned int det_id_type;

  template<typename T> class DetSetVector;
  
  /* a proxy to a variable size array of T belonging to
   * a "channel" identified by an 32bit id
   *
   * FIXME interface to be finalized once use-cases fully identified
   * 
   */
  template<typename T>
  class DetSet {
  public: 
    typedef DetSetVector<T> Container;
    typedef unsigned int size_type; // for persistency
    typedef unsigned int id_type;
    typedef T data_type;

    typedef std::vector<data_type> DataContainer;
    typedef data_type * iterator;
    typedef data_type const * const_iterator;

    typedef data_type value_type;
    typedef id_type key_type;

#ifdef USE_ATOMIC
    using RCU = std::shared_ptr<DataContainer>;
    // return the RCU in the Getter (shall be protected by onDemand) 
    //RCU  & rcu();
#endif

    
    inline
    DetSet() : m_id(0), m_data(nullptr), m_offset(-1), m_size(0){}
    inline
    DetSet(id_type i, DataContainer const & idata, size_type ioffset, size_type isize) :
      m_id(i), m_data(&idata.front()), m_offset(ioffset), m_size(isize) {}
    
    inline
    DetSet(Container const & icont,
	   typename Container::Item const & item, bool update) :
      m_id(0), m_data(nullptr), m_offset(-1), m_size(0){
      set(icont,item, update);
    }

    //FIXME (it may confuse users as size_type is same type as id_type...)
    inline
    void set(Container const & icont,
	     typename Container::Item const & item, bool update=true);

    bool isValid() const { return m_offset>=0;}

    inline
    data_type & operator[](size_type i) {
      return data()[i];
    }
    
    inline
    data_type operator[](size_type i) const {
      return data()[i];
    }
    
    inline
    iterator begin() { return data();}

    inline
    iterator end() { return data()+m_size;}

    inline
    const_iterator begin() const { return data();}

    inline
    const_iterator end() const { return data()+m_size;}


    inline
    id_type id() const { return m_id;}
    
    inline
    id_type detId() const { return m_id;}
    
    inline
    size_type size() const { return m_size; }

    inline
    bool empty() const { return m_size==0;}
    
  private:
    data_type const * data() const {
      if(m_offset|m_size) assert(m_data);
      // return m_data ? (&((*m_data)[m_offset])) : 0;
      return m_data ? m_data + m_offset : nullptr;
    }

   data_type * data() {
     assert(m_data);
     return const_cast<data_type *>(m_data + m_offset);
     //     return const_cast<data_type *>(&((*m_data)[m_offset]));
    }
    
    id_type m_id;
    //DataContainer const * m_data;
    data_type const * m_data;
    int m_offset;
    size_type m_size;
#ifdef USE_ATOMIC
    RCU m_rcu;
#endif
  };
}

#endif // DataFormats_Common_DetSet_h
