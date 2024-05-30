<a name="3.0"></a>
## 3.0 (2018-07-25)


#### Features

* **feature:**  Support multi-channel images and publish more parameters ([d8b9a7e5]('https://github.com/fhkiel-mlaip/rfl'/commit/d8b9a7e584354c2b088d625cc9bda2b652fdcd41))



<a name="2.0"></a>
## 2.0 (2018-07-03)


#### Bug Fixes

* **opencl:**  Lock parallelized kernel calls and wait ([ef41da8d]('https://github.com/fhkiel-mlaip/rfl'/commit/ef41da8dc58040b591bfc1fa008701d43b65e360))
* **training:**  Limit negative samples only by already seen samples ([54a529bc]('https://github.com/fhkiel-mlaip/rfl'/commit/54a529bc91aa3e585a88055671c70cb01c6e8f4d))

#### Performance

* **opencl:**  Use 1D-based kernel invocation ([929ac1c7]('https://github.com/fhkiel-mlaip/rfl'/commit/929ac1c77451f104043f119512afa1e25258bdf2))

#### Features

* **api:**  Make create_gaussian_samples less pedantic ([bcf8eb03]('https://github.com/fhkiel-mlaip/rfl'/commit/bcf8eb03924fefb0a048b545c615351fb65041ca))
* **cli:**  Add the CLI, again ([26efbe7c]('https://github.com/fhkiel-mlaip/rfl'/commit/26efbe7cfeecf204e59683c2f7875a6d3162d6dc))
* **feature:**
  *  Support difference vectors from random start positions ([dd3643b1]('https://github.com/fhkiel-mlaip/rfl'/commit/dd3643b177dacc897f4212ce125ed8f1c2d76dfe))
  *  Use constant for out-of-bounds feature values ([09d14996]('https://github.com/fhkiel-mlaip/rfl'/commit/09d149961d177e6949cb758377c0e6f205d2ce0f))
* **training:**  Add possibility to adjust pre-smoothing sigma ([ebc4ff59]('https://github.com/fhkiel-mlaip/rfl'/commit/ebc4ff59f72ddf19e60eacc90999000c04410ef9))



<a name="1.0"></a>
## 1.0 (2017-01-22)


#### Breaking Changes

* **gpu:**  Use OpenCL to utilize GPU acceleration if possible ([8c403724](https://github.com/fhkiel-mlaip/rfl/commit/8c4037244232630f4c08f2d03b039f9a33c51e3b), breaks [#](https://github.com/fhkiel-mlaip/rfl/issues/))

#### Features

* **gpu:**  Use OpenCL to utilize GPU acceleration if possible ([8c403724](https://github.com/fhkiel-mlaip/rfl/commit/8c4037244232630f4c08f2d03b039f9a33c51e3b), breaks [#](https://github.com/fhkiel-mlaip/rfl/issues/))



<a name="0.4"></a>
## 0.4 (2016-11-13)


#### Bug Fixes

* **feature:**
  *  Drop BRIEF features (for now) ([651eafcb](https://github.com/fhkiel-mlaip/rfl/commit/651eafcb76a0e52fbe5a09462016a3a46b257c33), breaks [#](https://github.com/fhkiel-mlaip/rfl/issues/))
  *  Use chosen feature extractor ([94fe4c98](https://github.com/fhkiel-mlaip/rfl/commit/94fe4c988988fc21590429a16fb3a725f10ebf28))

#### Features

* **plot:**  Add command and API to show the used feature masks ([cf72846d](https://github.com/fhkiel-mlaip/rfl/commit/cf72846d51cc234713f5752845a4323012956ae4))
* **training:**  Train trees using batches of training images ([4919d6d8](https://github.com/fhkiel-mlaip/rfl/commit/4919d6d8c6a82166782ed3a8b22c3f4a18ae1f0e))

#### Breaking Changes

* **feature:**
  *  Use Cython to compute features and evaluate the tree ([4057e0cf](https://github.com/fhkiel-mlaip/rfl/commit/4057e0cfc66bc169b42f03a1023a428be45d7845), breaks [#](https://github.com/fhkiel-mlaip/rfl/issues/))
  *  Drop BRIEF features (for now) ([651eafcb](https://github.com/fhkiel-mlaip/rfl/commit/651eafcb76a0e52fbe5a09462016a3a46b257c33), breaks [#](https://github.com/fhkiel-mlaip/rfl/issues/))

#### Performance

* **feature:**  Use Cython to compute features and evaluate the tree ([4057e0cf](https://github.com/fhkiel-mlaip/rfl/commit/4057e0cfc66bc169b42f03a1023a428be45d7845), breaks [#](https://github.com/fhkiel-mlaip/rfl/issues/))



<a name="0.3"></a>
## 0.3 (2016-09-21)


#### Features

* **3d:**  Add basic support for 3D data ([2c300f7b](https://github.com/fhkiel-mlaip/rfl/commit/2c300f7b40db2d2fd7b17f25d8d6b17324d2cd91), closes [#4](https://github.com/fhkiel-mlaip/rfl/issues/4))
* **features:**  Support selection of voxel point pairing patterns ([57e602cb](https://github.com/fhkiel-mlaip/rfl/commit/57e602cb4525b14e9d4a816725dea46cb97cc2bc))



<a name="0.2"></a>
## 0.2 (2016-09-14)


#### Features

* **api:**  Add methods to de/serialize a RandomForestLocalize instance ([a33cec59](https://github.com/fhkiel-mlaip/rfl/commit/a33cec59a3d64555ac3cff4791d66c0bea2a630b), closes [#6](https://github.com/fhkiel-mlaip/rfl/issues/6))

#### Bug Fixes

* **regressor:**  Select negative samples from regressor predictions ([9db866fa](https://github.com/fhkiel-mlaip/rfl/commit/9db866fad39e1b1e4de07266f24d80834b549c10))



<a name="0.1"></a>
## 0.1 (2016-09-14)

* **Birth!**
