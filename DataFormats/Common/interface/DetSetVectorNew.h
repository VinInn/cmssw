#ifndef DataFormats_Common_DetSetVectorNew_h
#define DataFormats_Common_DetSetVectorNew_h

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
// #include "DataFormats/Common/interface/DetSet.h"  // to get det_id_type
#include "DataFormats/Common/interface/DetSetNew.h"
#include "DataFormats/Common/interface/traits.h"


#include <boost/iterator_adaptors.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/any.hpp>
#include <memory>
#include "FWCore/Utilities/interface/GCC11Compatibility.h"
#include "FWCore/Utilities/interface/HideStdSharedPtrFromRoot.h"

#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
#define USE_ATOMIC
// #warning using atomic
#endif

#ifdef  USE_ATOMIC
#include <atomic>
#include <thread>
#include <memory>
#endif

#include<vector>
#include <cassert>

namespace edm { namespace refhelper { template<typename T> struct FindForNewDetSetVector; } }

//FIXME remove New when ready
namespace edmNew {
  typedef uint32_t det_id_type;

  namespace dslv {
    template< typename T> class LazyGetter;

  }

  /* transient component of DetSetVector
   * for pure conviniency of dictionary declaration
   */
  namespace dstvdetails {

    void errorFilling();
    void notSafe();
    void errorIdExists(det_id_type iid);
    void throw_range(det_id_type iid);

    struct DetSetVectorTrans {
      DetSetVectorTrans(): filling(false){}
#ifndef USE_ATOMIC
      mutable bool filling;
  private:
      DetSetVectorTrans& operator=(const DetSetVectorTrans&){return *this;}
      DetSetVectorTrans(const DetSetVectorTrans&){}
  public:
#else
      DetSetVectorTrans& operator=(const DetSetVectorTrans&) = delete;
      DetSetVectorTrans(const DetSetVectorTrans&) = delete;
      DetSetVectorTrans(DetSetVectorTrans&&) = default;
      DetSetVectorTrans& operator=(DetSetVectorTrans&&) = default;
      mutable std::atomic<bool> filling;
#endif
      boost::any getter;


      void swap(DetSetVectorTrans& rh) {
	// better no one is filling...
        assert(filling==false); assert(rh.filling==false);	
	//	std::swap(filling,rh.filling);
	std::swap(getter,rh.getter);
      }

      typedef unsigned int size_type; // for persistency
      typedef unsigned int id_type;

      struct Item {
	Item(id_type i=0, int io=-1, size_type is=0) : id(i), offset(io), size(is){}
	id_type id;
#ifdef USE_ATOMIC
	//	Item(Item&&)=default;	Item & operator=(Item&& rh)=default;
	Item(Item const & rh)  noexcept :
	id(rh.id),offset(rh.offset.load(std::memory_order_acquire)),size(rh.size) {
	}
	Item & operator=(Item const & rh) noexcept {
	  id=rh.id;offset=rh.offset.load(std::memory_order_acquire);size=rh.size; return *this;
	}
	Item(Item&& rh)  noexcept :
	id(std::move(rh.id)),offset(rh.offset.load(std::memory_order_acquire)),size(std::move(rh.size)) {
	}
	Item & operator=(Item&& rh) noexcept {
	  id=std::move(rh.id);offset=rh.offset.load(std::memory_order_acquire);size=std::move(rh.size); return *this;
	}
	std::atomic<int> offset;
#else
	int offset;
#endif
	size_type size;

	bool isValid() const { return offset>=0;}
	bool operator<(Item const &rh) const { return id<rh.id;}
	operator id_type() const { return id;}
      };

#ifdef USE_ATOMIC
      bool ready() {
        bool expected=false;
        if (!filling.compare_exchange_strong(expected,true))  errorFilling();
        return true;
      }
#else
      bool ready() {return true;}
#endif

    };

   }

  /** an optitimized container that linearized a "map of vector".
   *  It corresponds to a set of variable size array of T each belonging
   *  to a "Det" identified by an 32bit id
   *
   * FIXME interface to be finalized once use-cases fully identified
   *
   * although it is sorted internally it is strongly adviced to
   * fill it already sorted....
   *
   */
  template<typename T>
  class DetSetVector  : private dstvdetails::DetSetVectorTrans {
  public:
    typedef dstvdetails::DetSetVectorTrans Trans;
    typedef Trans::Item Item;
    typedef unsigned int size_type; // for persistency
    typedef unsigned int id_type;
    typedef T data_type;
    typedef edmNew::DetSetVector<T> self;
    typedef edmNew::DetSet<T> DetSet;
    typedef dslv::LazyGetter<T> Getter;
    // FIXME not sure make sense....
    typedef DetSet value_type;
    typedef id_type key_type;


    typedef std::vector<Item> IdContainer;
    typedef std::vector<data_type> DataContainer;
    typedef typename IdContainer::iterator IdIter;
    typedef typename std::vector<data_type>::iterator DataIter;
    typedef std::pair<IdIter,DataIter> IterPair;
    typedef typename IdContainer::const_iterator const_IdIter;
    typedef typename std::vector<data_type>::const_iterator const_DataIter;
    typedef std::pair<const_IdIter,const_DataIter> const_IterPair;

    typedef typename edm::refhelper::FindForNewDetSetVector<data_type>  RefFinder;
    
    struct IterHelp {
      typedef DetSet result_type;
      //      IterHelp() : v(0),update(true){}
      IterHelp() : v(0),update(false){}
      IterHelp(DetSetVector<T> const & iv, bool iup) : v(&iv), update(iup){}
      
      result_type & operator()(Item const& item) const {
	detset.set(*v,item,update);
	return detset;
      } 
    private:
      DetSetVector<T> const * v;
      mutable result_type detset;
      bool update;
    };
    
    typedef boost::transform_iterator<IterHelp,const_IdIter> const_iterator;
    typedef std::pair<const_iterator,const_iterator> Range;

    /* fill the lastest inserted DetSet
     */
    class FastFiller {
    public:
      typedef typename DetSetVector<T>::data_type value_type;
      typedef typename DetSetVector<T>::id_type key_type;
      typedef typename DetSetVector<T>::id_type id_type;
      typedef typename DetSetVector<T>::size_type size_type;

#ifdef USE_ATOMIC
      static DetSetVector<T>::Item & dummy() {
	static  DetSetVector<T>::Item d; return d;
      }
      FastFiller(DetSetVector<T> & iv, id_type id, bool isaveEmpty=false) : 
	v(iv), item(v.ready()? v.push_back(id): dummy()),saveEmpty(isaveEmpty) {
        if (v.onDemand()) dstvdetails::notSafe();
      }

      FastFiller(DetSetVector<T> & iv, typename DetSetVector<T>::Item & it, bool isaveEmpty=false) : 
	v(iv), item(it), saveEmpty(isaveEmpty) {
	if (v.onDemand()) dstvdetails::notSafe();
	if(v.ready()) item.offset = int(v.m_data.size());

      }
      ~FastFiller() {
	if (!saveEmpty && item.size==0) {
	  v.pop_back(item.id);
	}
	assert(v.filling==true);
	v.filling.store(false,std::memory_order_release);

      }
      
#endif
      
      void abort() {
	v.pop_back(item.id);
	saveEmpty=true; // avoid mess in destructor
      }

      void reserve(size_type s) {
	v.m_data.reserve(item.offset+s);
      }
      
      
      void resize(size_type s) {
	v.m_data.resize(item.offset+s);
	item.size=s;
      }

      id_type id() const { return item.id;}
      size_type size() const { return item.size;}
      bool empty() const { return item.size==0;}

      data_type & operator[](size_type i) {
	return 	v.m_data[item.offset+i];
      }
      DataIter begin() { return v.m_data.begin()+ item.offset;}
      DataIter end() { return begin()+size();}

      void push_back(data_type const & d) {
	v.m_data.push_back(d);
	item.size++;
      }
#ifndef CMS_NOCXX11
      void push_back(data_type && d) {
        v.m_data.push_back(std::move(d));
        item.size++;
      }
#endif

      data_type & back() { return v.m_data.back();}
      
    private:
      DetSetVector<T> & v;
      typename DetSetVector<T>::Item & item;
      bool saveEmpty;
    };

    /* fill on demand a given  DetSet
     */
    class TSFastFiller {
    public:
      typedef typename DetSetVector<T>::data_type value_type;
      typedef typename DetSetVector<T>::id_type key_type;
      typedef typename DetSetVector<T>::id_type id_type;
      typedef typename DetSetVector<T>::size_type size_type;

#ifdef USE_ATOMIC
      static DetSetVector<T>::Item & dummy() {
        static  DetSetVector<T>::Item d; return d;
      }
      TSFastFiller(DetSetVector<T> & iv, id_type id) :
        v(iv), item(v.ready()? v.push_back(id): dummy()) { assert(v.filling==true); v.filling = false;}

      TSFastFiller(DetSetVector<T> & iv, typename DetSetVector<T>::Item & it) : 
	v(iv), item(it) {

      }
      ~TSFastFiller() {
	bool expected=false;
	while (!v.filling.compare_exchange_weak(expected,true,std::memory_order_acq_rel))  { expected=false; nanosleep(0,0);}
	int offset = v.m_data.size();
	if (v.m_data.capacity()<offset+lv.size()) {
	  auto newcap = 2*v.m_data.capacity()+lv.size();
	  // chek if there are readers..
	  if (v.onDemand() && (!v.rcu().unique())) { //there is at least a reader around do rcu
	    auto & old = v.rcu();
	    (*old).swap(v.m_data);
	    v.m_data.clear(); v.m_data.reserve(newcap);
	    std::copy((*old).begin(),(*old).end(),std::back_inserter(v.m_data));
	    assert(offset=v.m_data.size());
	    old.reset(new typename DetSetVector<T>::DataContainer());  // old data will be deleted when reader "release" lock
	  }else {
	    v.m_data.reserve(newcap);
	  }
	}
	std::move(lv.begin(), lv.end(), std::back_inserter(v.m_data));
	item.size=lv.size();
	item.offset = offset; 

	assert(v.filling==true);
	v.filling.store(false,std::memory_order_release);
      }
      
#endif
      
      void abort() {
	lv.clear();
      }

      void reserve(size_type s) {
	lv.reserve(s);
      }

      void resize(size_type s) {
	lv.resize(s);
      }

      id_type id() const { return item.id;}
      size_type size() const { return lv.size();}
      bool empty() const { return lv.empty();}

      data_type & operator[](size_type i) {
	return 	lv[i];
      }
      DataIter begin() { return lv.begin();}
      DataIter end() { return lv.end();}

      void push_back(data_type const & d) {
	lv.push_back(d);
      }
#ifndef CMS_NOCXX11
      void push_back(data_type && d) {
        lv.push_back(std::move(d));
      }
#endif

      data_type & back() { return v.m_data.back();}
      
    private:
      std::vector<T> lv;
      DetSetVector<T> & v;
      typename DetSetVector<T>::Item & item;
    };



    friend class FastFiller;
    friend class TSFastFiller;
    friend class edmNew::DetSet<T>;

    class FindForDetSetVector : public std::binary_function<const edmNew::DetSetVector<T>&, unsigned int, const T*> {
    public:
        typedef FindForDetSetVector self;
        typename self::result_type operator()(typename self::first_argument_type iContainer, typename self::second_argument_type iIndex) {
#ifdef USE_ATOMIC
	  bool expected=false;
	  while (!iContainer.filling.compare_exchange_weak(expected,true,std::memory_order_acq_rel))  { expected=false; nanosleep(0,0);}
#else
	  iContainer.filling = true;
#endif
	  typename self::result_type item =  &(iContainer.m_data[iIndex]);
	  assert(iContainer.filling==true);
	  iContainer.filling = false;
	  return item;
        }
    };
    friend class FindForDetSetVector;

    explicit DetSetVector(int isubdet=0) :
      m_subdetId(isubdet) {}

    DetSetVector(std::shared_ptr<dslv::LazyGetter<T> > iGetter, const std::vector<det_id_type>& iDets,
		 int isubdet=0);


    ~DetSetVector() {
      // delete content if T is pointer...
    }

#ifdef USE_ATOMIC
    // default or delete is the same...
    DetSetVector& operator=(const DetSetVector&) = delete;
    DetSetVector(const DetSetVector&) = delete;
    DetSetVector(DetSetVector&&) = default;
    DetSetVector& operator=(DetSetVector&&) = default;
#else
  private:
    DetSetVector& operator=(const DetSetVector&){return *this;}
    DetSetVector(const DetSetVector&){}
  public:
#endif

    bool onDemand() const { return !getter.empty();}


#ifdef USE_ATOMIC
    using RCU = std::shared_ptr<DataContainer>;
    // return the RCU in the Getter (shall be protected by onDemand) 
    RCU  & rcu();
    RCU const & rcu() const;
#endif




    void swap(DetSetVector & rh) {
      DetSetVectorTrans::swap(rh);
      std::swap(m_subdetId,rh.m_subdetId);
      std::swap(m_ids,rh.m_ids);
      std::swap(m_data,rh.m_data);
    }
    
    void swap(IdContainer & iic, DataContainer & idc) {
      std::swap(m_ids,iic);
      std::swap(m_data,idc);
    }
    
    void reserve(size_t isize, size_t dsize) {
      m_ids.reserve(isize);
      m_data.reserve(dsize);
    }
    
    void shrink_to_fit() {
#ifndef CMS_NOCXX11
      clean();   
      m_ids.shrink_to_fit();
      m_data.shrink_to_fit();
#endif
    }

    void resize(size_t isize, size_t dsize) {
      m_ids.resize(isize);
      m_data.resize(dsize);
    }

    void clean() {
#ifndef CMS_NOCXX11
      m_ids.erase(std::remove_if(m_ids.begin(),m_ids.end(),[](Item const& m){ return 0==m.size;}),m_ids.end());
#endif
    }
    
    // FIXME not sure what the best way to add one cell to cont
    DetSet insert(id_type iid, data_type const * idata, size_type isize) {
      Item & item = addItem(iid,isize);
      m_data.resize(m_data.size()+isize);
      std::copy(idata,idata+isize,m_data.begin()+item.offset);
      return DetSet(*this,item,false);
    }
    //make space for it
    DetSet insert(id_type iid, size_type isize) {
      Item & item = addItem(iid,isize);
      m_data.resize(m_data.size()+isize);
      return DetSet(*this,item,false);
    }

    // to be used with a FastFiller
    Item & push_back(id_type iid) {
      return addItem(iid,0);
    }

    // remove last entry (usually only if empty...)
    void pop_back(id_type iid) {
      const_IdIter p = findItem(iid);
      if (p==m_ids.end()) return; //bha!
      // sanity checks...  (shall we throw or assert?)
      if ((*p).size>0&& (*p).offset>-1 && 
	  m_data.size()==(*p).offset+(*p).size)
	m_data.resize((*p).offset);
      m_ids.erase( m_ids.begin()+(p-m_ids.begin()));
    }

  private:

    Item & addItem(id_type iid,  size_type isize) {
      Item it(iid,size_type(m_data.size()),isize);
      IdIter p = std::lower_bound(m_ids.begin(),
				  m_ids.end(),
				  it);
      if (p!=m_ids.end() && !(it<*p)) dstvdetails::errorIdExists(iid);
#ifndef CMS_NOCXX11
      return *m_ids.insert(p,std::move(it));
#else
      return *m_ids.insert(p,it);
#endif
    }



  public:


    //---------------------------------------------------------
    
    bool exists(id_type i) const  {
      return  findItem(i)!=m_ids.end(); 
    }
        
    bool isValid(id_type i) const {
      const_IdIter p = findItem(i);
      return p!=m_ids.end() && (*p).offset!=-1;
    }

    /*
    DetSet operator[](id_type i) {
      const_IdIter p = findItem(i);
      if (p==m_ids.end()) what???
      return DetSet(*this,p-m_ids.begin());
    }
    */

    
    DetSet operator[](id_type i) const {
      const_IdIter p = findItem(i);
      if (p==m_ids.end()) dstvdetails::throw_range(i);
      return DetSet(*this,*p,true);
    }
    
    // slow interface
    //    const_iterator find(id_type i, bool update=true) const {
    const_iterator find(id_type i, bool update=false) const {
      const_IdIter p = findItem(i);
      return (p==m_ids.end()) ? end() :
	boost::make_transform_iterator(p,
				       IterHelp(*this,update));
    }

    // slow interface
    const_IdIter findItem(id_type i) const {
      std::pair<const_IdIter,const_IdIter> p =
	std::equal_range(m_ids.begin(),m_ids.end(),Item(i));
      return (p.first!=p.second) ? p.first : m_ids.end();
    }
    
    //    const_iterator begin(bool update=true) const {
    const_iterator begin(bool update=false) const {
      return  boost::make_transform_iterator(m_ids.begin(),
					     IterHelp(*this,update));
    }

    //    const_iterator end(bool update=true) const {
    const_iterator end(bool update=false) const {
      return  boost::make_transform_iterator(m_ids.end(),
					     IterHelp(*this,update));
    }
    

    // return an iterator range (implemented here to avoid dereference of detset)
    template<typename CMP>
      //    Range equal_range(id_type i, CMP cmp, bool update=true) const {
    Range equal_range(id_type i, CMP cmp, bool update=false) const {
      std::pair<const_IdIter,const_IdIter> p =
	std::equal_range(m_ids.begin(),m_ids.end(),i,cmp);
      return  Range(boost::make_transform_iterator(p.first,IterHelp(*this,update)),
		    boost::make_transform_iterator(p.second,IterHelp(*this,update))
		    );
    }
    
    int subdetId() const { return m_subdetId; }

    bool empty() const { return m_ids.empty();}


    size_type dataSize() const { return m_data.size(); }
    
    size_type size() const { return m_ids.size();}
    
    //FIXME fast interfaces, not consistent with associative nature of container....

    data_type operator()(size_t cell, size_t frame) const {
      return m_data[m_ids[cell].offset+frame];
    }
    
    data_type const * data(size_t cell) const {
      return &m_data[m_ids[cell].offset];
    }
    
    size_type detsetSize(size_t cell) const { return  m_ids[cell].size; }

    id_type id(size_t cell) const {
      return m_ids[cell].id;
    }

    Item const & item(size_t cell) const {
      return m_ids[cell];
    }

    //------------------------------

    IdContainer const & ids() const { return m_ids;}
    DataContainer const & data() const { return  m_data;}


    void update(Item const & item) const {
      const_cast<self*>(this)->updateImpl(const_cast<Item&>(item));
    }
   
    //Used by ROOT storage
    CMS_CLASS_VERSION(10)

  private:

    void updateImpl(Item & item);
    
  private:
    // subdetector id (as returned by  DetId::subdetId())
    int m_subdetId;
    
    IdContainer m_ids;
    DataContainer m_data;
    
  };
  
 namespace dslv {
    template< typename T>
    class LazyGetter {
    public:
      virtual ~LazyGetter() {}
      virtual void fill(typename DetSetVector<T>::TSFastFiller&) = 0;

#ifdef USE_ATOMIC
      LazyGetter() : rcu(new typename DetSetVector<T>::DataContainer()){}
      std::shared_ptr<typename DetSetVector<T>::DataContainer> rcu;
#endif
    };
  }
  

#ifdef USE_ATOMIC
  template<typename T>
  inline typename DetSetVector<T>::RCU  & DetSetVector<T>::rcu() {
    return (*boost::any_cast<std::shared_ptr<Getter> >(&getter))->rcu;
  }
  template<typename T>
  inline typename DetSetVector<T>::RCU  const & DetSetVector<T>::rcu() const {
    return (*boost::any_cast<std::shared_ptr<Getter> >(&getter))->rcu;
  }
#endif
    

  template<typename T>
  inline DetSetVector<T>::DetSetVector(std::shared_ptr<Getter> iGetter, 
				       const std::vector<det_id_type>& iDets,
				       int isubdet):  
    m_subdetId(isubdet) {
    getter=iGetter;

    m_ids.reserve(iDets.size());
    det_id_type sanityCheck = 0;
    for(std::vector<det_id_type>::const_iterator itDetId = iDets.begin(), itDetIdEnd = iDets.end();
	itDetId != itDetIdEnd;
	++itDetId) {
      assert(sanityCheck < *itDetId && "vector of det_id_type was not ordered");
      sanityCheck = *itDetId;
      m_ids.push_back(*itDetId);
    }
  }

  template<typename T>
  inline void DetSetVector<T>::updateImpl(Item & item) {
    // no getter or already updated
/*
    if (getter.empty()) assert(item.offset>=0);
    if (item.offset!=-1 || getter.empty() ) return;
    item.offset = int(m_data.size());
    FastFiller ff(*this,item,true);
    (*boost::any_cast<std::shared_ptr<Getter> >(&getter))->fill(ff);
*/
    if (getter.empty()) { assert(item.offset>=0); return;}
#ifdef USE_ATOMIC
    int expected = -1;
    if (item.offset.compare_exchange_strong(expected,-2,std::memory_order_acq_rel)) {
      assert(item.offset==-2); 
      {
	TSFastFiller ff(*this,item);
	(*boost::any_cast<std::shared_ptr<Getter> >(&getter))->fill(ff);
      }
      assert(item.offset>=0);
    }
#endif
  }
 
  
  template<typename T>
  inline void DetSet<T>::set(DetSetVector<T> const & icont,
			     typename Container::Item const & item, bool update) {
#ifdef USE_ATOMIC
    // if an item is being updated we wait (cannot do RCU at this very moment)
    if (update)icont.update(item);
    while(item.offset.load(std::memory_order_acquire)<-1) nanosleep(0,0);
    
    bool expected=false;
    while (!icont.filling.compare_exchange_weak(expected,true,std::memory_order_acq_rel))  { expected=false; nanosleep(0,0);}
    if(icont.onDemand()) m_rcu = icont.rcu();
#endif
    m_data=&icont.data().front();
#ifdef USE_ATOMIC
    icont.filling.store(false,std::memory_order_release);
    //  if(icont.onDemand()) assert(&(*m_rcu).front() == m_data); // only if not empty which is rare here
#endif
    m_id=item.id; 
    m_offset = item.offset; 
    m_size=item.size;
  }
  
}

#include "DataFormats/Common/interface/Ref.h"
#include <boost/mpl/assert.hpp>
#include <boost/type_traits/is_same.hpp>

//specialize behavior of edm::Ref to get access to the 'Det'
namespace edm {
    /* Reference to an item inside a new DetSetVector ... */
    namespace refhelper {
        template<typename T>
            struct FindTrait<typename edmNew::DetSetVector<T>,T> {
                typedef typename edmNew::DetSetVector<T>::FindForDetSetVector value;
            };
    }
    /* ... as there was one for the original DetSetVector*/

    /* Probably this one is not that useful .... */
    namespace refhelper {
        template<typename T>
            struct FindSetForNewDetSetVector : public std::binary_function<const edmNew::DetSetVector<T>&, unsigned int, edmNew::DetSet<T> > {
                typedef FindSetForNewDetSetVector<T> self;
                typename self::result_type operator()(typename self::first_argument_type iContainer, typename self::second_argument_type iIndex) {
                    return &(iContainer[iIndex]);
                }
            };

        template<typename T>
            struct FindTrait<edmNew::DetSetVector<T>, edmNew::DetSet<T> > {
                typedef FindSetForNewDetSetVector<T> value;
            };
    }
    /* ... implementation is provided, just in case it's needed */
}

namespace edmNew {
   //helper function to make it easier to create a edm::Ref to a new DSV
  template<class HandleT>
  edm::Ref<typename HandleT::element_type, typename HandleT::element_type::value_type::value_type>
  makeRefTo(const HandleT& iHandle,
             typename HandleT::element_type::value_type::const_iterator itIter) {
    BOOST_MPL_ASSERT((boost::is_same<typename HandleT::element_type, DetSetVector<typename HandleT::element_type::value_type::value_type> >));
    typename HandleT::element_type::size_type index = (itIter - &iHandle->data().front());
    assert(index>=0); assert(index<iHandle->data().size());
    return edm::Ref<typename HandleT::element_type,
	       typename HandleT::element_type::value_type::value_type>
	      (iHandle,index);
  }

  template<class HandleT, typename DS>
  edm::Ref<typename HandleT::element_type, typename HandleT::element_type::value_type::value_type>
  makeRefTo(const HandleT& iHandle, DS const & ds,
             typename HandleT::element_type::value_type::const_iterator itIter) {
    BOOST_MPL_ASSERT((boost::is_same<typename HandleT::element_type, DetSetVector<typename HandleT::element_type::value_type::value_type> >));
    typename HandleT::element_type::size_type index = ds.index(itIter); 
    return edm::Ref<typename HandleT::element_type,
	       typename HandleT::element_type::value_type::value_type>
	      (iHandle,index);
  }
}


#include "DataFormats/Common/interface/ContainerMaskTraits.h"

namespace edm {
   template<typename T>
   class ContainerMaskTraits<edmNew::DetSetVector<T> > {
     public:
        typedef T value_type;

        static size_t size(const edmNew::DetSetVector<T>* iContainer) { return iContainer->dataSize();}
        static unsigned int indexFor(const value_type* iElement, const edmNew::DetSetVector<T>* iContainer) {
           unsigned int index = iElement-&(iContainer->data().front());
	   assert(index>=0); assert(index<iContainer->data().size());
	   return index;
        }
   };
}


#ifdef  USE_ATOMIC
#undef  USE_ATOMIC
#endif

#endif
  
