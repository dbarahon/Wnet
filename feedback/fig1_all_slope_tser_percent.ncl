

;************************************************
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl" 
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
;load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"   
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/contrib/cd_string.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/contrib/time_axis_labels.ncl"
load  "/gpfsm/dnb34/dbarahon/bin/ncl_ncar/my_functions.ncl"
;************************************************

 
begin
 
   mod_tit =  "Norm W StdDev Anomaly"
   maxlogN =  1.5
    minlogN  = 0.0
   last_yr =  2021

levs =(/250., 500., 900./)
 
 tit  =  "fig_all_slope_tser_percent_combined" 
  lv_off =10
  wks_type = "eps"  
  wks_type@wkPaperSize = "A4"
  
  
  ;  colormp = "wh-bl-gr-ye-re" 
 ; colormp =  "BlAqGrYeOrReVi200"
 ; colormp =  "BlAqGrYeOrRe"
 ; colormp ="WhiteBlueGreenYellowRed"
 colormp =  "BlueWhiteOrangeRed"
 ;colormp = "MPL_coolwarm"
  ;colormp =  "precip32_17lev"
  ;colormp="inferno"
  
  
  ;cmp = span_named_colors((/ "white", "red" /), 64)  
  




  
     wks   = gsn_open_wks (wks_type, tit)                  ; open workstation

 
 
 
  plots1 = new(10,graphic)
  plots2 = new(10,graphic)
  plots_sig = new(10,graphic)                                    ; create a plot array
  format_time = "%N/%D %H:%M:%S"
  fontsz = 0.030
  
   gsn_define_colormap(wks, colormp)   
   
   
;************************************************
; settings
;************************************************
  res                      = True                   ; plot mods desired   
  res@cnLevelSelectionMode = "ManualLevels"         ; manually select levels
  res@cnFillOn             = True                   ; turn on color
  res@cnLinesOn            = False                  ; no contour lines
  res@cnLineLabelsOn       = False                  ; no contour lines
  res@gsnSpreadColors      = True                   ; use full range of colors
  res@lbLabelAutoStride    = True                  ; optimal labels  
 
  res@gsnDraw              = False                   ; don't draw yet
  res@gsnFrame             = False                   ; don't advance frame yet
  res@tiMainFontHeightF    = 16
  res@tiMainFont           = "Helvetica-Bold"
  
  res@lbLabelBarOn          = False
  res@lbTitlePosition = "Bottom"
  
  ;res@tmXBLabelsOn         = False              ; do not draw bottom labels
  ;res@tmXBOn               = False              ; no bottom tickmarks
  res@tmXBLabelFontHeightF = fontsz                 ; resize tick labels
  res@tmYLLabelFontHeightF = fontsz  
  res@gsnStringFontHeightF  =fontsz
  res@mpCenterLonF          = 0    
  res@cnInfoLabelOn = False                     ; turn off cn info label
  res@gsnMajorLatSpacing = 30
  res@gsnMajorLonSpacing = 60 
  res@lbLabelFontHeightF =fontsz
  res@pmLabelBarOrthogonalPosF = .2
  res@lbBoxEndCapStyle = "TriangleBothEnds"

  res@tmXBLabelFontHeightF = fontsz                 ; resize tick labels
  res@tmYLLabelFontHeightF = fontsz    
  res@gsnStringFontHeightF  =fontsz
  res@tiYAxisFontHeightF =fontsz*1.2
  res@tiXAxisFontHeightF =fontsz*1.3
  res@cnFillMode = "CellFill"
  res@mpGeophysicalLineThicknessF  =  2.5
  res@lbTitleFontHeightF    = fontsz*1.3
   
  res@gsnRightString = ""
   
  
  ;******** SW resources 
   
  res_sw =  res
  opt = True
  opt@NumColorsInRange = (/5,1,1,20/)
  
  cmap = read_colormap_file("MPL_RdYlBu") 
  res_sw@cnFillPalette =cmap(::-1,:); "wh-bl-gr-ye-re"; "GMT_no_green" ; span_named_colors((/ "white", "blue", "yellow",  "red" /), opt)   
  
   
  res_sw@cnSmoothingOn =  True
  res_sw@cnLineLabelFormat = "%1.1e"
  l_h = ispan (-70,20,5)*0.01    
  l_l =  10.0^l_h
  ;print(l_l)
  res_sw@cnLevelSelectionMode = "ExplicitLevels"
  res_sw@cnExplicitLabelBarLabelsOn =  True
  res_sw@cnLevels = l_l
  levsc =  sprintf("%1.2f", l_l)
  res_sw@lbLabelStrings  =levsc ;     = (/0.01, 0.05, 0.1, 0.5, 1, 5/)   
  res_sw@cnSpanFillPalette =True
  res_sw@gsnSpreadColors      = False
  
  
 
  ;;;=========line resources 
     
    resxy=res  
  resxy@xyLineThicknessF =4.0          

   delete(resxy@xyMarkerColors)
   delete(resxy@xyLineColors)
   resxy@xyMarkLineModes     = "Lines" 
   resxy@gsMarkerIndex = 2
   resxy@xyMarkerColors = (/"blue"/)
   resxy@xyLineColors           = (/"blue"/)
  ; resxy@trYMaxF            = 6.
  ; resxy@trYMinF            = -6. 
   resxy@trXMaxF            = 2022
   resxy@tiYAxisString   = "~F8~s~F~~B~W~N~ anom. (%)" ;
   resxy@tiXAxisString =  ""
   resxy@vpWidthF         = 0.62
   resxy@vpHeightF        = 0.29
   resxy@tmYLFormat    = "f" 
     ;--------------model---------
   resxy2 =  resxy
   resxy2@xyLineColors           = (/"black"/)
   resxy2@xyDashPatterns     = (/1/)
   

 ;======= Hatching resources
   
   res2                  = True
   res2@gsnDraw              = False                   ; don't draw yet
   res2@gsnFrame             = False                   ; don't advance frame yet
   res2@cnMonoFillColor = True
   res2@cnFillOn             = True                 ; turn on color
   res2@cnMinLevelValF       = -1.             
   res2@cnMaxLevelValF       = 1.
   res2@cnLevelSpacingF   = 1.
   res2@lbLabelBarOn          = False
   res2@cnFillDotSizeF       = 0.002 
   res2@cnInfoLabelOn = False   
   
   res2@cnMonoFillPattern = False              ; want multiple patterns 
   res2@cnMonoFillScale   = False              ; want patterns w/ diff densities
  
   res2@cnFillPatterns    = (/ -1,  17,  17  \    ; 3=slant lines
                          ,-1 ,-1 ,-1  \    ; -1=transparent
                          , 17, 17, 17, 17, 17/) ; stipling
   
   res2@cnFillScales      = (/0.65, 0.65, 2.\   ; line density
                          , 1., 1., 1.  \   ; default [no op]
                          ,0.7,0.6,0.5,0.40,0.3/) ; stiple density
                           
 
   res2@cnLinesOn            = False                 ; no contour lines
   res2@cnLineLabelsOn       = False                 ; no contour lines
   
   res2@cnLevelSelectionMode = "AutomaticLevels"
   res2@gsnRightString = ""
   res2@gsnLeftString = ""
   ;res2@trGridType = "TriangularMesh"
   ;res2@tfDoNDCOverlay = False
    tcrit  =  2.02 ; two tailed 95%  
       
; read in and plot
;************************************************  
  
  
  fil =  "../combined_ERA5_M2/stats/SWclim_ann_Np.nc4.nc"
  data = addfile(fil,"r")
  Wclim = data->SWclim(:, :, :)
  Wclim =  Wclim;  
  Wclim!0 = "p"
  Wclim!1 = "lat"  
  Wclim!2 = "lon"
  Wclim&p@units = "hPa"                                    
  Wclim&lat@units = "degrees north" 
  Wclim&lon@units=  "degrees east"
  printVarSummary(Wclim)
  lat=Wclim&lat
  clat2d = cos(lat*0.01745329)
  lab = (/"(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"/)
  order = (/0, 2, 4, 1, 3, 5/)
  ;===========SW_plot======
  
  do i =  0, 2 
      lv  = levs(i) ; up one from surface
      lv1 =  lv-lv_off 
      lv2 =  lv+lv_off

      sl =  Wclim({lv1:lv2}, :, :)
      uavg2 = sl(lat|:, lon|:, p|0)
      uavg2= dim_avg(sl(lat|:, lon|:, p|:))
      mn =  wgt_areaave(uavg2, clat2d, 1.0, 0) 
      res_sw@gsnRightString = sprintf("%1.3f", mn) + " m s~S~-1~N~"
      res_sw@gsnLeftString =  lab(i) + " " + sprintf("%3.0f", lv)+ " hPa "
      
      if (i .eq. 2) then 
      	res_sw@lbLabelBarOn          = True        
        res_sw@lbTitleString    = "~F8~s~F~~B~W~N~ (m s~S~-1~N~)"; "sigmaW m/s"
        
      end if
      
      plots1(order(i)) = gsn_csm_contour_map(wks,uavg2, res_sw)  ; create plot 


  end do   
  
  delete(uavg2)
  ;==========slope_plot
  
    folder  = "../combined_ERA5_M2/"
    data = addfile(folder + "stats/SWanom_monthly_Np.nc4","r")
    
    data_norm = addfile(folder + "stats/SWanom_norm_monthly_Np.nc4","r")
    var_norm  =  "SWanom_norm"
    res@cnMinLevelValF       = -0.6             
    res@cnMaxLevelValF       = 0.6
    res@cnLevelSpacingF   = 0.1 
    var  =  "SWanom"
             
    Wall = data->$var$
    Wall!0 = "time"
    Wall!1 = "lev"
    Wall!2 = "lat"      
    Wall!3 = "lon" 
    
    
    printVarSummary(Wall)
    
    Wnorm =  Wall
    Wnorm = data_norm->$var_norm$
    
    tmod    = yyyymm_time(1980, last_yr, "integer") ; contributed.ncl 
    t_mod =  yyyymm_to_yyyyfrac(tmod, 0.0)
    ;print(t_mod)
    ti =  t_mod
   
    x  = 0
  
          
    do i = 0, 2
        x =  order(i+3)
        lv  = levs(i); middle atm
        lv1 =  lv-lv_off 
        lv2 =  lv+lv_off
        sl_u=Wall(:, {lv1:lv2}, :, :) 
        uavg=sl_u(lev|0, time|:, lat|:, lon|:)
        uavg=dim_avg_Wrap( sl_u(time|:, lat|:, lon|:, lev|:))
        
        sl =  Wclim({lv1:lv2}, :, :)
        SWclim_l = sl(lat|:, lon|:, p|0)
        SWclim_l= dim_avg(sl(lat|:, lon|:, p|:))
        
        
        txx =  uavg(lat|:, lon|:, time|0)
        dimy = dimsizes(txx) 
        df  = new ((/dimy(0),dimy(1)/) , integer)
        rc  = uavg(lat|:, lon|:, time|0)
        
        rc = regcoef(ti,uavg(lat|:, lon|:, time|:), txx, df)
        ;copy_VarCoords(uavg(lat|:, lon|:, time|0), rc) 
        rc!0   = "lat"    ; name dimensions
        rc!1   = "lon"
        rc&lat = uavg&lat    ; assign coordinate values to named dimensions
        rc&lon = uavg&lon             
        
        rc =  100.*rc/SWclim_l
        
        printVarSummary(rc) 
        
        nm =  wgt_areaave(rc, clat2d, 1.0, 0)   
        
       ; printVarSummary(txx)
        tavg2 =  txx 

        tavg2 =  where(txx .gt. tcrit, 0.5, -0.5) 	
        

        res@gsnLeftString = lab(i+3) + " " + sprintf("%3.0f", lv)+ " hPa " 
        res@gsnRightString = ""; sprintf("%1.3f", nm)  + " % yr~S~-1~N~"
       
        if (i .eq. 2) then 
      	   res@lbLabelBarOn          = True
           res@lbTitleString    = "~F8~s~F~~B~W~N~ trend (% yr~S~-1~N~)"; "sigmaW m/s"
        end if
        
        plots1(x) = gsn_csm_contour_map(wks,rc, res)  ; create plot

        plot_sig = gsn_csm_contour(wks,tavg2, res2)  ; create plot
        ;plots1(i) = gsn_csm_contour_map(wks,tavg2, res2) 
          ;plot_sig(x) = gsn_contour_shade(plot_sig(i), 0, 0.8, res2)

         overlay(plots1(x), plot_sig)
         
         delete(uavg)
         delete(rc)
         delete(txx)
         delete(tavg2)
         
    end do
    
  


  ;========time series plot
  
    restick = True
    ; restick@ttmFormat = "%N%y"
    ; time_axis_labels(timeaxis,resxy,restick) ; call the formatting procedure

    folder  = "./"
    data = addfile(folder + "../combined_ERA5_M2/stats/SWanom_monthly_Np.nc4","r")

 
   do i = 0, 2
    lv  = levs(i); middle atm
    lv1 =  lv-lv_off 
    lv2 =  lv+lv_off
	sl_u=Wall(:, {lv1:lv2}, :, :) 
    

    uavg=sl_u(time|:, lat|:, lon|:, lev|0)
    uavg=dim_avg_Wrap(sl_u(time|:, lat|:, lon|:, lev|:))
    uavg2x=wgt_areaave(uavg, clat2d, 1.0, 0) ; 
   
    sl =  Wclim({lv1:lv2}, :, :)
    SWclim_l = sl(lat|:, lon|:, p|0)
    SWclim_l= dim_avg(sl(lat|:, lon|:, p|:))
    
    Wmean = wgt_areaave(SWclim_l, clat2d, 1.0, 0)
        
    uavg2 =  100.*uavg2x/Wmean
            
    printVarSummary(uavg2)    

    ; slope
    resxy@gsnLeftString = lab(i+6) + " " + sprintf("%3.0f", lv)+ " hPa" 
    rc   = regline_stats (ti,uavg2)  
    yReg =  rc@Yest   ;

    rc@b95 =  rc@b95  
    rc@b =  rc@b                
    rcint = rc@b95(1) - rc@b(1)         
   ; resxy@gsnRightString = "m=" +  sprintf("%3.3f", rc@b(1)) + "~F34~1~F~" + sprintf("%3.3f",rcint) ;+ "cms~S~-1~N~yr~S~-1" 
    resxy@gsnRightString = sprintf("%3.3f", rc@b(1)) + "% yr~S~-1"                
    
    if (i .gt. 1) then 
    	resxy@tiXAxisString = "time" 
    end if    
    plots2(i)  = gsn_csm_xy (wks,ti,uavg2,resxy)          
    
    plot_reg  = gsn_csm_xy (wks,ti,yReg,resxy2)          
    overlay(plots2(i), plot_reg ) 

  
    delete(uavg)
    delete(uavg2)
    
    end do

 
        ;***********************************************
        ; create combined plot
        ;************************************************
        
         ;======== combined plot resources =========
   
        resP            = True                         ; modify the panel plot
        resP@lbLabelAutoStride    = True

        resP@gsnPanelLabelBar    = False 
        resP@lbLabelFontHeightF  = 0.015
        resP@gsnMaximize      = False                       ; use full page
        resP@gsnPanelXWhiteSpacePercent = 2.
        resP@lbTitleFontHeightF    = fontsz/2.
        resP@lbTitleString    = "~F8~s~F~~B~W~N~ trend (% yr~S~-1~N~)"; "sigmaW m/s"

        resP@lbTitlePosition = "Bottom"
        resP@pmLabelBarOrthogonalPosF = -.04
        resP@gsnPaperMargin= 1
        resP@lbBoxEndCapStyle = "TriangleBothEnds"
        resP@gsnPanelScalePlotIndex =  1
        resP@gsnMaximize =  False

        ; resP@txString   =  "Normalized sigmaW anom  " + d ; mod_tit + "  " + d

         ;gsn_panel(wks,plots2, (/3, 1/),resP)             ; now draw as one plot
        
         

         pres =  resP
         resP2 =  resP
         resP2@pmPanelDebug = True
         resP@gsnPanelYWhiteSpacePercent=2.5
         resP2@gsnPanelYWhiteSpacePercent=0.5
         
         resP2@gsnPanelBottom = 0.2
         resP2@gsnPanelTop = 0.8
         
         ;pres@gsnPanelLeft   = 0.66
         ;pres@gsnPanelRight  = 0.66

         panel_two_sets(wks,plots1,plots2,(/3,2/),(/3,1/),resP,resP2,pres)
         
         ;trimPNG(tit)



end
