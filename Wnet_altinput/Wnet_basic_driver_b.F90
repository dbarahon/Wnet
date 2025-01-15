program test_Wnet
  use Wnet
  implicit none
!!!only use with no standardization
  integer, parameter :: dp = selected_real_kind(15, 307)
  integer, parameter :: Ns = 100
  integer, parameter :: NF = 11
  real(dp) :: input(Ns, NF)
  real(dp) :: output(Ns)
  integer :: i

 
  ! Example input
  !input = reshape([ (real(i, dp), i=1, Ns * NF) ], shape=[Ns, NF])

  
   ! Example input (random values for testing)
  do i = 1, Ns
      input(i, :) = (i-50)*0.1_dp
  end do


  print *, "Running Wnet with Ns =", Ns
  call run_Wnet(input, output, Ns)

  print *, "Output:"
  do i = 1, Ns
    print *, "Sample ", (i-50)*0.1_dp, ": ", output(i)
  end do
end program test_Wnet
